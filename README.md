# Binance Semi-Auto Trader

> **AVISO:** Aplicación educativa de paper trading. No involucra dinero real.  
> Las señales generadas NO son predicciones ni garantías de rentabilidad.  
> El trading implica riesgo significativo de pérdidas.

---

## Stack

- Python 3.11 + FastAPI + Uvicorn
- SQLAlchemy 2 + SQLite
- Pydantic Settings
- Pandas / NumPy
- WebSockets
- binance-connector (SDK oficial)

## Modos de funcionamiento

| Modo | Descripción |
|------|-------------|
| `signal` | Solo genera señales, sin órdenes |
| `paper` | Simula operaciones con precios reales (default) |
| `demo` | Usa Binance Spot Demo Mode con confirmación manual |

El modo `live` **no está implementado** y es rechazado al arrancar.

---

## Instalación (macOS)

### 1. Requisitos previos

```bash
# Instalar Homebrew si no está instalado
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar Python 3.11
brew install python@3.11

# Verificar versión
python3.11 --version
```

### 2. Clonar el repositorio

```bash
git clone https://github.com/aguirreeg96-cpu/binance-.git
cd binance-
```

### 3. Crear entorno virtual

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tu editor preferido
# En modo paper/signal no se necesitan API keys
nano .env
```

Variables mínimas para modo paper:
```
TRADING_MODE=paper
TRADING_SYMBOL=BTCUSDT
TRADING_INTERVAL=1h
```

### 6. Ejecutar la aplicación

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Abrir en el navegador: http://localhost:8000/docs

### 7. Ejecutar los tests

```bash
pytest tests/ -v
```

---

## Estructura del proyecto

```
app/
├── main.py              # FastAPI app, lifespan, endpoints de sistema
├── config.py            # Pydantic Settings (carga .env, validaciones de seguridad)
├── database.py          # SQLAlchemy engine + session factory
├── models/              # Modelos ORM (9 tablas)
│   ├── candle.py
│   ├── signal.py
│   ├── paper_account.py
│   ├── position.py
│   ├── order.py
│   ├── trade.py
│   ├── strategy_config.py
│   ├── daily_risk_state.py
│   └── system_event.py
└── schemas/
    └── common.py        # Enums tipados compartidos
tests/
├── test_config.py       # Tests de configuración y guards de seguridad
└── test_models.py       # Tests de modelos y esquema de base de datos
```

## Etapas de implementación

- [x] Etapa 1: Estructura y configuración
- [ ] Etapa 2: Datos históricos (klines REST)
- [ ] Etapa 3: Indicadores técnicos (EMA, RSI, ATR, volumen)
- [ ] Etapa 4: Estrategia de señales
- [ ] Etapa 5: Backtesting
- [ ] Etapa 6: Paper trading
- [ ] Etapa 7: Panel web
- [ ] Etapa 8: Binance Demo Mode
- [ ] Etapa 9: Confirmación manual
- [ ] Etapa 10: Tests completos y documentación

---

## Seguridad

- Las claves API nunca se escriben en el código.
- Las URLs de producción de Binance son rechazadas activamente.
- El modo `live` no existe en esta aplicación.
- Los secretos se enmascaran en todos los logs.
- `.env` está en `.gitignore` y nunca se commitea.
