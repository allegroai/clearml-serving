import json
import os
from argparse import ArgumentParser, FileType

from .serving_service import ServingService


def restore_state(args):
    session_state_file = os.path.expanduser('~/.clearml_serving.json')
    # noinspection PyBroadException
    try:
        with open(session_state_file, 'rt') as f:
            state = json.load(f)
    except Exception:
        state = {}

    args.id = getattr(args, 'id', None) or state.get('id')
    return args


def store_state(args, clear=False):
    session_state_file = os.path.expanduser('~/.clearml_serving.json')
    if clear:
        state = {}
    else:
        state = {str(k): str(v) if v is not None else None
                 for k, v in args.__dict__.items() if not str(k).startswith('_') and k not in ('command', )}
    # noinspection PyBroadException
    try:
        with open(session_state_file, 'wt') as f:
            json.dump(state, f, sort_keys=True)
    except Exception:
        pass


def cmd_triton(args):
    if not args.id and not args.name:
        raise ValueError("Serving service must have a name, use --name <service_name>")

    if args.id and not args.project and not args.name:
        a_serving = ServingService(task_id=args.id)
    else:
        a_serving = ServingService(task_project=args.project, task_name=args.name, engine_type='triton')
        args.id = a_serving.get_id()

    if args.endpoint:
        print("Nvidia Triton Engine ID: {} - Adding serving endpoint: \n".format(args.id) +
              ("model-project: '{}', model-name: '{}', model-tags: '{}', config-file: '{}'".format(
                  args.model_project or '',
                  args.model_name or '',
                  args.model_tags or '',
                  args.config or '') if not args.model_id else
               "model-id: '{}', config-file: '{}'".format(args.model_id or '', args.config or '')))

    if not args.endpoint and (args.model_project or args.model_tags or args.model_id or args.model_name):
        raise ValueError("Serving endpoint must be provided, add --endpoint <endpoint_name>")

    if args.endpoint:
        a_serving.add_model_serving(
            serving_url=args.endpoint,
            model_project=args.model_project,
            model_name=args.model_name,
            model_tags=args.model_tags,
            model_ids=[args.model_id] if args.model_id else None,
            config_file=args.config,
            max_versions=args.versions,
        )

    a_serving.serialize(force=True)
    store_state(args)


def cmd_launch(args):
    print('Launching Serving Engine: service: {}, queue: {}'.format(args.id, args.queue))

    if not args.id:
        raise ValueError("Serving service must specify serving service ID, use --id <service_id>")

    a_serving = ServingService(task_id=args.id)

    if a_serving.get_engine_type() not in ('triton',):
        raise ValueError("Error, serving engine type \'{}\' is not supported".format(a_serving.get_engine_type()))

    # launch services queue
    a_serving.launch(queue_name=args.service_queue)
    # launch engine
    a_serving.launch_engine(queue_name=args.queue)


def cli(verbosity):
    title = 'clearml-serving - CLI for launching ClearML serving engine'
    print(title)
    parser = ArgumentParser(prog='clearml-serving', description=title)
    parser.add_argument('--debug', action='store_true', help='Print debug messages')
    subparsers = parser.add_subparsers(help='Serving engine commands', dest='command')

    # create the launch command
    parser_launch = subparsers.add_parser('launch', help='Launch a previously configured serving service')
    parser_launch.add_argument(
        '--id', default=None, type=str,
        help='Specify a previously configured service ID, if not provided use the last created service')
    parser_launch.add_argument(
        '--queue', default=None, type=str, required=True,
        help='Specify the clearml queue to be used for the serving engine server')
    parser_launch.add_argument(
        '--service-queue', default='services', type=str,
        help='Specify the service queue to be used for the serving service, default: services queue')
    parser_launch.set_defaults(func=cmd_launch)

    # create the parser for the "triton" command
    parser_trt = subparsers.add_parser('triton', help='Nvidia Triton Serving Engine')
    parser_trt.add_argument(
        '--id', default=None, type=str,
        help='Add configuration to running serving session, pass serving Task ID, '
             'if passed ignore --name / --project')
    parser_trt.add_argument(
        '--name', default=None, type=str,
        help='Give serving service a name, should be a unique name')
    parser_trt.add_argument(
        '--project', default='DevOps', type=str,
        help='Serving service project name, default: DevOps')
    parser_trt.add_argument(
        '--endpoint', required=False, type=str,
        help='Serving endpoint, one per model, unique ')
    parser_trt.add_argument(
        '--versions', type=int,
        help='Serving endpoint, support multiple versions, '
             'max versions to deploy (version number always increase). Default (no versioning).')
    parser_trt.add_argument(
        '--config', required=False, type=FileType,
        help='Model `config.pbtxt` file, one per model, order matching with models')
    parser_trt.add_argument(
        '--model-id', type=str,
        help='(Optional) Model ID to deploy, if passed model-project/model-name/model-tags are ignored')
    parser_trt.add_argument(
        '--model-project', type=str, help='Automatic model deployment and upgrade, select model project (exact match)')
    parser_trt.add_argument(
        '--model-name', type=str, help='Automatic model deployment and upgrade, select model name (exact match)')
    parser_trt.add_argument(
        '--model-tags', nargs='*', type=str,
        help='Automatic model deployment and upgrade, select model name tags to include, '
             'model has to have all tags to be deployed/upgraded')
    parser_trt.set_defaults(func=cmd_triton)

    args = parser.parse_args()
    verbosity['debug'] = args.debug
    args = restore_state(args)

    if args.command:
        args.func(args)
    else:
        parser.print_help()


def main():
    verbosity = dict(debug=False)
    try:
        cli(verbosity)
    except KeyboardInterrupt:
        print('\nUser aborted')
    except Exception as ex:
        print('\nError: {}'.format(ex))
        if verbosity.get('debug'):
            raise ex
        exit(1)


if __name__ == '__main__':
    main()
