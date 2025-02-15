import uvicorn

from app import create_app

app = create_app()

if __name__ == '__main__':
    uvicorn.run("main:app", headers=[("server", "zaklabs_api")], host="0.0.0.0", port=8004, 
                reload=True, log_level="debug")
