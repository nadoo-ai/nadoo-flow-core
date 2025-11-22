"""
Backend Registry
Factory for creating workflow backend instances
"""

from typing import Dict, Type, Callable
from .protocol import IWorkflowBackend
from .nadoo_backend import NadooBackend


class BackendRegistry:
    """Registry for workflow execution backends

    This class implements the Factory pattern for creating backend instances.
    Users can register custom backends and select which one to use.

    Example:
        # Use default native backend
        backend = BackendRegistry.create("native")

        # Register custom backend
        BackendRegistry.register("custom", MyCustomBackend)
        backend = BackendRegistry.create("custom")

        # Set default backend
        BackendRegistry.set_default("custom")
    """

    _backends: Dict[str, Type[IWorkflowBackend] | Callable[[], IWorkflowBackend]] = {
        "native": NadooBackend,
    }

    _default_backend: str = "native"

    @classmethod
    def register(
        cls,
        name: str,
        backend_class: Type[IWorkflowBackend] | Callable[[], IWorkflowBackend]
    ):
        """Register a new backend

        Args:
            name: Name to register the backend under
            backend_class: Backend class or factory function

        Example:
            BackendRegistry.register("langchain", LangChainBackend)
        """
        cls._backends[name] = backend_class

    @classmethod
    def unregister(cls, name: str):
        """Unregister a backend

        Args:
            name: Name of the backend to unregister

        Raises:
            ValueError: If trying to unregister the nadoo backend
        """
        if name == "native":
            raise ValueError("Cannot unregister the default 'native' backend")

        if name in cls._backends:
            del cls._backends[name]

    @classmethod
    def create(cls, name: str | None = None) -> IWorkflowBackend:
        """Create a backend instance

        Args:
            name: Name of the backend to create. If None, uses default.

        Returns:
            Backend instance

        Raises:
            ValueError: If backend name is not registered

        Example:
            backend = BackendRegistry.create("native")  # Native Nadoo implementation
            backend = BackendRegistry.create()  # Uses default (native)
        """
        backend_name = name or cls._default_backend

        if backend_name not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"Backend '{backend_name}' not registered. "
                f"Available backends: {available}"
            )

        backend_class = cls._backends[backend_name]
        return backend_class()

    @classmethod
    def set_default(cls, name: str):
        """Set the default backend

        Args:
            name: Name of the backend to set as default

        Raises:
            ValueError: If backend name is not registered

        Example:
            BackendRegistry.set_default("native")
        """
        if name not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"Backend '{name}' not registered. "
                f"Available backends: {available}"
            )

        cls._default_backend = name

    @classmethod
    def get_default(cls) -> str:
        """Get the name of the default backend

        Returns:
            Name of the default backend
        """
        return cls._default_backend

    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered backends

        Returns:
            List of backend names
        """
        return list(cls._backends.keys())
