"""
This type stub file was generated by pyright.
"""

from ansys.aedt.core.generic.general_methods import pyaedt_function_handler

class MathUtils:
    """MathUtils is a utility class that provides methods for numerical comparisons and checks."""
    EPSILON = ...
    @staticmethod
    @pyaedt_function_handler()
    def is_zero(x: float, eps: float = ...) -> bool:
        """Check if a number is close to zero within a small epsilon tolerance.

        Parameters:
            x: float
                Number to check.
            eps : float
                Tolerance for the comparison. Default is ``EPSILON``.

        Returns:
            bool
                ``True`` if the number is numerically zero, ``False`` otherwise.

        """
        ...
    
    @staticmethod
    @pyaedt_function_handler()
    def is_close(a: float, b: float, relative_tolerance: float = ..., absolute_tolerance: float = ...) -> bool:
        """Whether two numbers are close to each other given relative and absolute tolerances.

        Parameters
        ----------
        a : float, int
            First number to compare.
        b : float, int
            Second number to compare.
        relative_tolerance : float
            Relative tolerance. The default value is ``1e-9``.
        absolute_tolerance : float
            Absolute tolerance. The default value is ``0.0``.

        Returns
        -------
        bool
            ``True`` if the two numbers are closed, ``False`` otherwise.
        """
        ...
    
    @staticmethod
    @pyaedt_function_handler()
    def is_equal(a: float, b: float, eps: float = ...) -> bool:
        """
        Return True if numbers a and b are equal within a small epsilon tolerance.

        Parameters:
            a: float
                First number.
            b: float
                Second number.
            eps : float
                Tolerance for the comparison. Default is ``EPSILON``.

        Returns:
            bool
                ``True`` if the absolute difference between a and b is less than epsilon, ``False`` otherwise.
        """
        ...
    
    @staticmethod
    @pyaedt_function_handler()
    def atan2(y: float, x: float) -> float:
        """Implementation of atan2 that does not suffer from the following issues:
        math.atan2(0.0, 0.0) = 0.0
        math.atan2(-0.0, 0.0) = -0.0
        math.atan2(0.0, -0.0) = 3.141592653589793
        math.atan2(-0.0, -0.0) = -3.141592653589793
        and returns always 0.0.

        Parameters
        ----------
        y : float
            Y-axis value for atan2.

        x : float
            X-axis value for atan2.

        Returns
        -------
        float

        """
        ...
    
    @staticmethod
    @pyaedt_function_handler()
    def is_scalar_number(x): # -> bool:
        """Check if a value is a scalar number (int or float).

        Parameters
        ----------
        x : object
            Value to check.

        Returns
        -------
        bool
            ``True`` if x is a scalar number, ``False`` otherwise.
        """
        ...
    
    @staticmethod
    @pyaedt_function_handler()
    def fix_negative_zero(value): # -> list[Any] | float:
        """Fix the negative zero.
        It supports lists (and nested lists).

        Parameters
        ----------
        value : float, List
            Value to be fixed.

        Returns
        -------
        float, List
            Fixed value.

        """
        ...
    


