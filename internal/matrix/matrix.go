package matrix

import "log"

// Умножение матриц
func Mul(a, b [][]float64) [][]float64 {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])

	if colsA != rowsB {
		log.Fatal("Invalid matrix dimensions for multiplication")
	}

	// Результат умножения будет размером rowsA x colsB
	result := make([][]float64, rowsA)
	for i := range result {
		result[i] = make([]float64, colsB)
	}

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

// Транспонирование матрицы
func Transpose(a [][]float64) [][]float64 {
	rowsA, colsA := len(a), len(a[0])
	result := make([][]float64, colsA)
	for i := range result {
		result[i] = make([]float64, rowsA)
	}

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			result[j][i] = a[i][j]
		}
	}
	return result
}

// Масштабирование матрицы
func Scale(a [][]float64, scale float64) [][]float64 {
	rows, cols := len(a), len(a[0])
	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = a[i][j] * scale
		}
	}
	return result
}
