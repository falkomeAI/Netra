try:
    from netra.integrations.bhashini import BhashiniClient
    __all__ = ["BhashiniClient"]
except ImportError:
    __all__ = []
