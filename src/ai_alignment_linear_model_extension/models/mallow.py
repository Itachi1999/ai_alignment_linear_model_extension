from __future__ import annotations

class MallowsModelUtil:
    def __init__(self, num_alternatives: int, phi: float) -> None:
        self.num_alternatives = num_alternatives
        self.phi = phi
        
    def expected_kendall_distance(
        self,
        phi: float,
    ) -> float:
        """
        Expected Kendall-tau distance from the reference ranking under
        the Mallows phi-model.

        For m alternatives:

            E[K] =
                m * phi / (1 - phi)
                - sum_{i=1}^m i * phi^i / (1 - phi^i)

        Boundary cases:
            phi = 0 -> E[K] = 0
            phi = 1 -> E[K] = m(m - 1) / 4
        """
        if self.num_alternatives < 1:
            raise ValueError("num_alternatives must be >= 1.")

        if not 0.0 <= phi <= 1.0:
            raise ValueError("phi must be in [0, 1].")

        if self.num_alternatives <= 1:
            return 0.0

        if phi == 0.0:
            return 0.0

        if phi == 1.0:
            return self.num_alternatives * (self.num_alternatives - 1) / 4.0

        expected_distance = (self.num_alternatives * phi) / (1.0 - phi)
        for i in range(1, self.num_alternatives + 1):
            phi_i = phi**i
            expected_distance -= (i * phi_i) / (1.0 - phi_i)
        return expected_distance
    
    def normalized_dispersion_from_phi(
        self,
        phi: float,
    ) -> float:
        """
        Convert the ordinary Mallows phi parameter to normalized dispersion.

            normalized_dispersion = 4 E[K] / (m (m - 1))
        """
        if self.num_alternatives < 1:
            raise ValueError("num_alternatives must be >= 1.")

        if not 0.0 <= phi <= 1.0:
            raise ValueError("phi must be in [0, 1].")

        if self.num_alternatives <= 1:
            return 0.0

        normalized_phi = (4.0 * self.expected_kendall_distance(phi)) / (self.num_alternatives * (self.num_alternatives - 1))
        return normalized_phi
    
    
    def phi_from_normalized_dispersion(
        self,
        dispersion: float,
        *,
        tolerance: float = 1e-9,
        max_iterations: int = 100,
    ) -> float:
        """
        Convert normalized Mallows dispersion to the ordinary Mallows phi.

        Finds phi in [0, 1] such that

            4 E_phi[K] / (m (m - 1)) = dispersion.

        Binary search is used because normalized dispersion is monotone
        increasing in phi.
        """
        if self.num_alternatives < 1:
            raise ValueError("num_alternatives must be >= 1.")

        if not 0.0 <= dispersion <= 1.0:
            raise ValueError("dispersion must be in [0, 1].")

        if self.num_alternatives <= 1 or dispersion == 0.0:
            return 0.0

        if dispersion == 1.0:
            return 1.0

        low = 0.0
        high = 1.0

        for _ in range(max_iterations):
            phi = (low + high) / 2.0
            current_dispersion = self.normalized_dispersion_from_phi(phi)
            if abs(current_dispersion - dispersion) <= tolerance:
                return phi

            if current_dispersion < dispersion:
                low = phi
            else:
                high = phi

        return (low + high) / 2.0