import json
import pydantic


class FastapiOp:
    """
    op = FastapiOp
    app = op.create_app()
    op.register_post_router(app, path='/test', model=model)

    if __name__ == '__main__':
        import uvicorn
        uvicorn.run(app)
    """

    @classmethod
    def from_configs(cls, configs: dict):
        """
        {path1: {path2: router_kwargs}}
        """
        app = cls.create_app()

        for path1, cfg in configs.items():
            sub_app = cls.create_sub_app()
            for path2, router_kwargs in path1.items():
                cls.register_post_router(sub_app, path2, **router_kwargs)

            app.include_router(sub_app, prefix=path1)

        return app

    @staticmethod
    def create_app():
        from fastapi import FastAPI

        return FastAPI()

    @staticmethod
    def create_sub_app():
        from fastapi import APIRouter

        return APIRouter()

    @staticmethod
    def register_post_router(
            app: 'FastAPI' or 'APIRouter',
            path,
            model,
            request_template: 'pydantic.BaseModel()' = None,
            response_template: 'pydantic.BaseModel()' = None,
            **post_kwargs
    ):
        request_template = dict if request_template is None else request_template
        response_template = dict if response_template is None else response_template

        @app.post(path, response_model=response_template)
        def post(data: request_template):
            if isinstance(data, pydantic.BaseModel):
                data = data.dict()
            ret = model(data, **post_kwargs)
            return ret


class FlaskOp:
    """
    op = FastapiOp
    app = op.create_app()
    op.register_post_router(app, path='/test', model=model)

    if __name__ == '__main__':
        app.run()
    """

    @classmethod
    def from_configs(cls, configs: dict):
        """
        {path1: {path2: router_kwargs}}
        """
        app = cls.create_app()

        for path1, cfg in configs.items():
            sub_app = cls.create_sub_app(path1)
            for path2, router_kwargs in path1.items():
                cls.register_post_router(sub_app, path2, **router_kwargs)

            app.register_blueprint(sub_app, url_prefix=path1)

        return app

    @staticmethod
    def create_app():
        from flask import Flask

        return Flask(__name__)

    @staticmethod
    def create_sub_app(name):
        from flask import Blueprint

        return Blueprint(name, __name__)

    @staticmethod
    def register_post_router(
            app: 'Flask' or 'Blueprint',
            path,
            model,
            request_template: 'pydantic.BaseModel()' = None,
            response_template: 'pydantic.BaseModel()' = None,
            **post_kwargs
    ):
        from flask import jsonify, request

        @app.post(path, endpoint=path)
        def post():
            data = request.get_data().decode('utf-8')
            data = json.loads(data)
            if request_template:
                data = request_template(**data)
                data = data.dict()

            ret = model(data, **post_kwargs)

            if response_template:
                ret = response_template(**ret)
                ret = ret.dict()

            return jsonify(ret)
