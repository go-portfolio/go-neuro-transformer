package utils

// Функция для генерации случайных данных (можно будет использовать для создания тестовых данных)
func RandomMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = float64(i + j) // Просто пример, можно заменить на случайные числа
		}
	}
	return matrix
}
