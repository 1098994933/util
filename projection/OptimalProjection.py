import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import unittest


class OptimalProjection:
    """
    OptimalProjection: get the best projection from multiple projections
    """

    def __init__(self, projections, y):
        """
        Initialize the class with projections and target values.

        :param projections: List of PCA projection results, each element is a 2D array.
        :param y: Target values corresponding to each point in the projections.
        """
        self.projections = projections
        self.y = y
        self.mean_y = np.mean(y)
        self.labels = np.array(y > self.mean_y).astype(int)
        self.best_x = None
        self.best_model = None
        self.best_rectangle = None

    def find_best_projection(self):
        """
        Find the best projection that maximizes the product of rectangle area and the number of class 1 samples inside.

        :return: Index of the best projection and its corresponding product value.
        """
        best_index = 0
        best_product = 0

        for i, projection in enumerate(self.projections):
            num_points = projection.shape[0]
            max_product = 0
            best_rect = None
            for x1 in range(num_points):
                for y1 in range(num_points):
                    for x2 in range(x1 + 1, num_points):
                        for y2 in range(y1 + 1, num_points):
                            # Define the rectangle
                            x_min = min(projection[x1, 0], projection[x2, 0])
                            x_max = max(projection[x1, 0], projection[x2, 0])
                            y_min = min(projection[y1, 1], projection[y2, 1])
                            y_max = max(projection[y1, 1], projection[y2, 1])
                            # Calculate the area of the rectangle
                            area = (x_max - x_min) * (y_max - y_min)
                            # Count the number of class 1 samples inside the rectangle
                            in_rectangle = (projection[:, 0] >= x_min) & (projection[:, 0] <= x_max) & (
                                    projection[:, 1] >= y_min) & (projection[:, 1] <= y_max)
                            num_class_1 = np.sum(self.labels[in_rectangle] == 1)
                            # Calculate the product
                            product = area * num_class_1
                            if product > max_product:
                                max_product = product
                                best_rect = (x_min, x_max, y_min, y_max)
            if max_product > best_product:
                best_product = max_product
                best_index = i
                self.best_x = projection
                self.best_rectangle = best_rect

        return best_index, best_product

    def plot_decision_boundary(self):
        """
        Plot the best rectangle on the best projection.
        """
        if self.best_x is None or self.best_rectangle is None:
            print("Please run find_best_projection() first.")
            return

        x_min, x_max, y_min, y_max = self.best_rectangle

        plt.scatter(self.best_x[:, 0], self.best_x[:, 1], c=self.labels, edgecolors='k')
        plt.title('Best Rectangle on Best Projection')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        # Plot the rectangle
        plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'r-')

        plt.show()


class TestOptimalProjection(unittest.TestCase):
    def test_find_best_projection(self):
        projections = [np.random.rand(10, 2) for _ in range(20)]
        y = np.random.rand(10)
        op = OptimalProjection(projections, y)
        best_index, best_product = op.find_best_projection()
        self.assertEqual(isinstance(best_index, int), True)
        self.assertEqual(isinstance(best_product, float), True)
        self.assertTrue(0 <= best_index < len(projections))
        self.assertTrue(0 <= best_product)
        best_projection = projections[best_index]
        print(best_index, best_product)
        print(best_projection)
        X = best_projection
        y = y
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.colorbar()
        plt.show()
        plt.scatter(X[:, 0], X[:, 1], c=op.labels)
        plt.colorbar()
        plt.show()


        

        op.plot_decision_boundary()


if __name__ == "__main__":
    unittest.main()
