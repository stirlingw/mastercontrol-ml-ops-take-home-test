from fastapi import FastAPI
from routers import predict_tabular_view, predict_text_view

app = FastAPI()
app.include_router(predict_tabular_view.predict_tabular_view, prefix="/predict_tabular", tags=["predict_tabular"])
app.include_router(predict_text_view.predict_text_view, prefix="/predict_text", tags=["predict_text"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
