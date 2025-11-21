package main

import (
	"fmt"

	"github.com/go-portfolio/go-neuro-transformer/internal/attention"
)

func main() {
	// Пример входных данных для внимания
	Q := [][]float64{
		{1, 0, 1},
		{0, 1, 0},
		{1, 1, 0},
	}

	K := [][]float64{
		{1, 0, 1},
		{0, 1, 0},
		{1, 1, 0},
	}

	V := [][]float64{
		{1, 0},
		{0, 1},
		{1, 1},
	}

	// Применяем механизм внимания
	attentionResult := attention.Attention(Q, K, V)

	// Печатаем результат
	fmt.Println("Attention result:")
	for _, row := range attentionResult {
		fmt.Println(row)
	}
}
