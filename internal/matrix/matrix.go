package matrix

import (
	"log"
	"math"
)

// Матрица
type Matrix [][]float64

// Умножение матриц
func MatrixMul(a, b Matrix) Matrix {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])
	if colsA != rowsB {
		log.Fatal("Invalid matrix dimensions")
	}
	result := make(Matrix, rowsA)
	for i := 0; i < rowsA; i++ {
		result[i] = make([]float64, colsB)
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

// Транспонирование матрицы
func Transpose(a Matrix) Matrix {
	rows, cols := len(a), len(a[0])
	result := make(Matrix, cols)
	for i := 0; i < cols; i++ {
		result[i] = make([]float64, rows)
		for j := 0; j < rows; j++ {
			result[i][j] = a[j][i]
		}
	}
	return result
}

// Масштабирование матрицы
func ScaleMatrix(a Matrix, scale float64) Matrix {
	rows, cols := len(a), len(a[0])
	result := make(Matrix, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = a[i][j] * scale
		}
	}
	return result
}

// Softmax
func Softmax(x Matrix) Matrix {
	rows, cols := len(x), len(x[0])
	result := make(Matrix, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
	}
	for i := 0; i < rows; i++ {
		var sumExp float64
		for j := 0; j < cols; j++ {
			result[i][j] = math.Exp(x[i][j])
			sumExp += result[i][j]
		}
		for j := 0; j < cols; j++ {
			result[i][j] /= sumExp
		}
	}
	return result
}
