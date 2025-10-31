"""
API интерфейс для модели Math NLP → SymPy
Использование:
    python main.py --mode api  # запуск FastAPI сервера
    python main.py --mode cli  # CLI интерфейс
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import *
from inference import MathTranslator

def cli_mode():
    """CLI интерфейс"""
    model_path = CHECKPOINTS_DIR / "best_model.pt"
    
    if not model_path.exists():
        print("❌ Модель не найдена. Сначала обучите модель: python train.py")
        return
    
    translator = MathTranslator(model_path, VOCAB_PATH)
    
    print("\n" + "="*60)
    print("🧮 Math NLP → SymPy Translator (CLI режим)")
    print("="*60)
    print("Введите математическую задачу на русском или английском")
    print("Для выхода введите 'exit' или 'quit'\n")
    
    while True:
        text = input(">>> ").strip()
        
        if text.lower() in ['exit', 'quit', 'выход']:
            print("До свидания!")
            break
        
        if not text:
            continue
        
        try:
            sympy_code = translator.translate(text)
            print(f"SymPy код: {sympy_code}\n")
        except Exception as e:
            print(f"❌ Ошибка: {e}\n")

def api_mode():
    """FastAPI сервер"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("❌ Установите FastAPI: pip install fastapi uvicorn")
        return
    
    model_path = CHECKPOINTS_DIR / "best_model.pt"
    
    if not model_path.exists():
        print("❌ Модель не найдена. Сначала обучите модель: python train.py")
        return
    
    translator = MathTranslator(model_path, VOCAB_PATH)
    
    app = FastAPI(title="Math NLP to SymPy API")
    
    class TranslateRequest(BaseModel):
        text: str
        max_length: int = 128
    
    class TranslateResponse(BaseModel):
        input: str
        output: str
    
    @app.post("/translate", response_model=TranslateResponse)
    def translate(request: TranslateRequest):
        try:
            sympy_code = translator.translate(request.text, request.max_length)
            return TranslateResponse(input=request.text, output=sympy_code)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    def health():
        return {"status": "ok"}
    
    print("\n🚀 Запуск API сервера...")
    print("📖 Документация: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    parser = argparse.ArgumentParser(description="Math NLP to SymPy")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli", help="Режим работы")
    args = parser.parse_args()
    
    if args.mode == "cli":
        cli_mode()
    else:
        api_mode()

if __name__ == "__main__":
    main()