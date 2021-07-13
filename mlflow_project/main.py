
from kedro.framework.session import KedroSession
from kedro.framework.startup import _add_src_to_path
from pathlib import Path


if __name__ == "__main__":


    _add_src_to_path(source_dir=Path.cwd()/'kedro_project/src',
                     project_path=Path.cwd()/'kedro_project')

    session = KedroSession.create(
        'leads', project_path=Path.cwd()/'kedro_project', env='base')

    project_context = session.load_context()

    project_context.run(pipeline_name='training')
    print('asdsfsdfsd')
   