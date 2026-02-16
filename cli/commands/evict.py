"""Evict command for NDIF - evict (remove) a model deployment."""

import click
import ray
import asyncio
import time

from ..lib.util import get_controller_actor_handle, get_model_key, notify_dispatcher
from ..lib.checks import check_prerequisites
from ..lib.session import get_env

from src.common.tracing import TracingContext, init_tracing, trace_span


@click.command()
@click.argument('checkpoint', required=False)
@click.option('--revision', default='main', help='Model revision/branch (default: main)')
@click.option('--all', 'evict_all', is_flag=True, help='Evict all deployments')
@click.option('--ray-address', default=None, help='Ray address (default: from NDIF_RAY_ADDRESS)')
@click.option('--broker-url', default=None, help='Broker URL (default: from NDIF_BROKER_URL)')
def evict(checkpoint: str, revision: str, evict_all: bool, ray_address: str, broker_url: str):
    """Evict (remove) a model deployment.

    CHECKPOINT: Model checkpoint (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
                Optional if using --all flag

    This command removes a running model deployment to free up resources.

    Examples:
        ndif evict gpt2
        ndif evict meta-llama/Llama-2-7b-hf --revision main
        ndif evict --all                               # Evict all deployments
        ndif evict openai-community/gpt2 --ray-address ray://localhost:10001
    """
    # Use session defaults if not provided
    ray_address = ray_address or get_env("NDIF_RAY_ADDRESS")
    broker_url = broker_url or get_env("NDIF_BROKER_URL")

    init_tracing("ndif-cli")

    evict_start = time.time()

    try:
        with trace_span(
            "cli.evict",
            attributes={
                "ndif.cli.checkpoint": checkpoint or "--all",
                "ndif.cli.revision": revision,
                "ndif.cli.evict_all": evict_all,
            },
        ) as span:
            # Check prerequisites silently
            check_prerequisites(broker_url=broker_url, ray_address=ray_address)

            # Validate arguments
            if not evict_all and not checkpoint:
                click.echo("✗ Error: Must provide either CHECKPOINT or --all flag", err=True)
                raise click.Abort()

            if evict_all and checkpoint:
                click.echo("✗ Error: Cannot use both CHECKPOINT and --all flag", err=True)
                raise click.Abort()

            # Connect to Ray (suppress verbose output)
            click.echo(f"Connecting to Ray at {ray_address}...")
            ray_connect_start = time.time()
            ray.init(address=ray_address, ignore_reinit_error=True, logging_level="error")
            ray_connect_time = time.time() - ray_connect_start
            click.echo(f"Connected to Ray ({ray_connect_time:.2f}s)")
            span.set_attribute("ndif.cli.ray_connect_time_s", ray_connect_time)

            # Get controller actor handle
            click.echo("Getting controller handle...")
            controller = get_controller_actor_handle()

            # Determine which model keys to evict
            if evict_all:
                # Get all deployed models from status
                click.echo("Fetching all deployments...")
                status_ref = controller.status.remote()
                status = ray.get(status_ref)

                # Extract model_keys from HOT deployments only
                deployments = status.get("deployments", {})
                model_keys = [
                    deployment_info["model_key"]
                    for deployment_info in deployments.values()
                    if "model_key" in deployment_info and deployment_info.get("deployment_level") == "HOT"
                ]

                if not model_keys:
                    click.echo("No deployments found to evict.")
                    return

            else:
                # Single model eviction
                click.echo(f"Generating model key for {checkpoint} (revision: {revision})...")

                # TODO: revision bug ("main" is not always the default revision)
                model_key_start = time.time()
                model_key = get_model_key(checkpoint, revision)
                model_key_time = time.time() - model_key_start
                click.echo(f"Model key generated ({model_key_time:.2f}s)")
                span.set_attribute("ndif.cli.model_key_time_s", model_key_time)

                model_keys = [model_key]

            span.set_attribute("ndif.model.keys", str(model_keys))
            span.set_attribute("ndif.cli.num_models", len(model_keys))

            # Evict the models
            if not evict_all:
                click.echo(f"Evicting {checkpoint}...")
            else:
                click.echo(f"Evicting {len(model_keys)} model(s)...")

            controller_start = time.time()
            trace_context = TracingContext.inject()
            object_ref = controller.evict.remote(
                model_keys=model_keys,
                trace_context=trace_context,
            )
            results = ray.get(object_ref)
            controller_time = time.time() - controller_start

            span.set_attribute("ndif.cli.controller_time_s", controller_time)

            # Display results
            total_gpus_freed = 0
            total_memory_freed = 0.0
            evicted_count = 0
            not_found_count = 0

            for model_key, result in results.items():
                # Extract repo_id for display from deployments dict if available
                if evict_all:
                    repo_id = next(
                        (d["repo_id"] for d in deployments.values() if d.get("model_key") == model_key),
                        model_key
                    )
                else:
                    repo_id = checkpoint

                if result["status"] == "not_found":
                    if len(model_keys) == 1:
                        click.echo(f"✗ {repo_id} not found")
                    else:
                        click.echo(f"  ✗ {repo_id}: not found")
                    not_found_count += 1
                else:
                    if len(model_keys) == 1:
                        click.echo(f"✓ Evicted {repo_id}")
                        click.echo(f"  GPUs freed: {result['freed_gpus']}")
                        click.echo(f"  Memory freed: {round(result['freed_memory_gbs'], 4)} GB")
                    else:
                        click.echo(f"  ✓ {repo_id}: evicted")

                    total_gpus_freed += result['freed_gpus']
                    total_memory_freed += result['freed_memory_gbs']
                    evicted_count += 1

                    # Notify dispatcher about eviction
                    asyncio.run(notify_dispatcher(broker_url, "evict", model_key))

            span.set_attribute("ndif.cli.evicted_count", evicted_count)
            span.set_attribute("ndif.cli.not_found_count", not_found_count)
            span.set_attribute("ndif.cli.total_gpus_freed", total_gpus_freed)
            span.set_attribute("ndif.cli.total_memory_freed_gbs", total_memory_freed)

            # Summary (only for multiple models)
            if len(model_keys) > 1:
                click.echo()
                if evicted_count > 0:
                    click.echo(f"✓ Successfully evicted {evicted_count} model(s)")
                    click.echo(f"  Total GPUs freed: {total_gpus_freed}")
                    click.echo(f"  Total memory freed: {round(total_memory_freed, 4)} GB")

                if not_found_count > 0:
                    click.echo(f"✗ {not_found_count} model(s) not found")

            click.echo(f"Controller evict call took {controller_time:.2f}s")
            total_time = time.time() - evict_start
            span.set_attribute("ndif.cli.total_time_s", total_time)
            click.echo(f"Total CLI evict time: {total_time:.2f}s")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()
