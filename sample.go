package main

import(
	"fmt"
	// "sync"
	"math"
	"github.com/gonum/matrix/mat64"
)

func Sigmoid(x float64) (y float64) {
	return 1.0 / (1.0 + math.Exp(-1*x))
}

func main(){
	m := mat64.NewDense(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		
	})

	m2 := mat64.NewDense(3, 3, []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		
	})
	m3 := mat64.NewDense(3, 3, nil)
	m3.MulElem(m, m2)
	fmt.Println(mat64.Formatted(m3))
	// var wg sync.WaitGroup
	// for i:=0; i<20; i++{
	// 	wg.Add(1)
	// 	go func(num int){
	// 		defer wg.Done()
	// 		process(num)
			        
	// 	}(i)
		    
	// }
	// wg.Wait()
	// e = time.Now()
	// fmt.Printf("処理完了 : %v Seconds\n", (e.Sub(s)).Seconds())
	
}
// func process(num int){
// 	fmt.Printf("%d 番目の処理開始\n", num)
// 	time.Sleep(1 * time.Second)
// 	fmt.Printf("%d 番目の処理終了\n", num)
	
// }
