import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Генерирует синусоидальные позиционные кодирования.
    Возвращает np.ndarray формы (seq_length, d_model) dtype=float64.
    """
    # 1. Инициализируем матрицу нулей
    pe = np.zeros((seq_length, d_model), dtype=np.float64)
    
    # 2. Вектор позиций: форма (seq_length, 1) для broadcasting
    pos = np.arange(seq_length, dtype=np.float64).reshape(-1, 1)
    
    # 3. Вектор обратных частот: форма (1, d_model//2)
    # i принимает значения 0, 2, 4, ..., d_model-2 (это 2i из формулы)
    # div_term = 1 / 10000^(2i/d_model) = exp(-2i/d_model * ln(10000))
    i = np.arange(0, d_model, 2, dtype=np.float64)
    div_term = np.exp(i * (-np.log(10000.0) / d_model)).reshape(1, -1)
    
    # 4. Заполняем чётные колонки sin(), нечётные cos()
    # pos * div_term автоматически даёт матрицу (seq_length, d_model//2)
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    
    return pe