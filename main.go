// Copyright 2022 The Hyperion Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/disintegration/gift"
	"github.com/nfnt/resize"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tc128"
)

const (
	testImage = "images/image01.png"
	blockSize = 8
	netWidth  = 3 * blockSize * blockSize
	hiddens   = 5
	scale     = 1
)

var (
	// NeuralCompressMode use neural compress mode
	NeuralCompressMode = flag.Bool("neural", false, "neural compress mode")
	// KroneckerCompressMode use kronecker compress mode
	KroneckerCompressMode = flag.Bool("kronecker", false, "kronecker compress mode")
)

func neuralCompress() {
	rand.Seed(7)

	file, err := os.Open(testImage)
	if err != nil {
		panic(err)
	}

	info, err := file.Stat()
	if err != nil {
		panic(err)
	}
	name := info.Name()
	name = name[:strings.Index(name, ".")]

	input, _, err := image.Decode(file)
	if err != nil {
		panic(err)
	}
	file.Close()

	width, height := input.Bounds().Max.X, input.Bounds().Max.Y
	width, height = width/scale, height/scale
	input = resize.Resize(uint(width), uint(height), input, resize.NearestNeighbor)
	width -= width % blockSize
	height -= height % blockSize
	bounds := image.Rect(0, 0, width, height)
	g := gift.New(
		gift.Crop(bounds),
	)
	cropped := image.NewRGBA64(bounds)
	g.Draw(cropped, input)
	input = cropped

	size := width * height / (blockSize * blockSize)

	file, err = os.Create(name + ".png")
	if err != nil {
		panic(err)
	}

	err = png.Encode(file, input)
	if err != nil {
		panic(err)
	}
	file.Close()

	file, err = os.Create(name + ".jpg")
	if err != nil {
		panic(err)
	}

	err = jpeg.Encode(file, input, &jpeg.Options{Quality: 45})
	if err != nil {
		panic(err)
	}
	file.Close()

	set := tc128.NewSet()
	set.Add("layer1", netWidth, hiddens)
	set.Add("bias1", hiddens)
	set.Add("layer2", hiddens, netWidth)
	set.Add("bias2", netWidth)
	set.Add("image", netWidth, size/8)
	set.Add("full", netWidth, size)

	random128 := func(a, b float64) complex128 {
		return complex((b-a)*rand.Float64()+a, (b-a)*rand.Float64()+a)
	}

	for i := range set.Weights[:4] {
		w := set.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, complex128(0))
			}
		} else {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, random128(-1, 1)/complex(math.Sqrt(float64(w.S[0])), 0))
			}
		}
	}

	for n := 0; n < size/8; n++ {
		i := rand.Intn(width/blockSize) * blockSize
		j := rand.Intn(height / blockSize * blockSize)
		for y := 0; y < blockSize; y++ {
			for x := 0; x < blockSize; x++ {
				r, g, b, _ := input.At(i+x, j+y).RGBA()
				w := set.Weights[4]
				w.X = append(w.X, complex(float64(r)/0xFFFF, 0))
				w.X = append(w.X, complex(float64(g)/0xFFFF, 0))
				w.X = append(w.X, complex(float64(b)/0xFFFF, 0))
			}
		}
	}

	for j := 0; j < height; j += blockSize {
		for i := 0; i < width; i += blockSize {
			for y := 0; y < blockSize; y++ {
				for x := 0; x < blockSize; x++ {
					r, g, b, _ := input.At(i+x, j+y).RGBA()
					w := set.Weights[5]
					w.X = append(w.X, complex(float64(r)/0xFFFF, 0))
					w.X = append(w.X, complex(float64(g)/0xFFFF, 0))
					w.X = append(w.X, complex(float64(b)/0xFFFF, 0))
				}
			}
		}
	}

	l1 := tc128.Sigmoid(tc128.Add(tc128.Mul(set.Get("layer1"), set.Get("full")), set.Get("bias1")))
	l2 := tc128.Abs(tc128.Add(tc128.Mul(set.Get("layer2"), l1), set.Get("bias2")))
	cost := tc128.Avg(tc128.Quadratic(set.Get("full"), l2))

	eta, iterations := .0001+.0001i, 2048
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := complex128(0)
		start := time.Now()
		set.Zero()

		o := 0
		for n := 0; n < size/8; n++ {
			i := rand.Intn(width/blockSize) * blockSize
			j := rand.Intn(height / blockSize * blockSize)
			for y := 0; y < blockSize; y++ {
				for x := 0; x < blockSize; x++ {
					r, g, b, _ := input.At(i+x, j+y).RGBA()
					w := set.Weights[4]
					w.X[o] = complex(float64(r)/0xFFFF, 0)
					w.X[o+1] = complex(float64(g)/0xFFFF, 0)
					w.X[o+2] = complex(float64(b)/0xFFFF, 0)
					o += 3
				}
			}
		}

		total += tc128.Gradient(cost).X[0]
		norm := 0.0
		for _, p := range set.Weights[:4] {
			for _, d := range p.D {
				norm += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm = math.Sqrt(norm)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}

		for _, p := range set.Weights[:4] {
			for l, d := range p.D {
				p.X[l] -= eta * d * complex(scaling, 0)
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: cmplx.Abs(total)})
		fmt.Println(i, cmplx.Abs(total), time.Now().Sub(start))
		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}

	coded := image.NewRGBA64(input.Bounds())
	l1 = tc128.TanH(tc128.Add(tc128.Mul(set.Get("layer1"), set.Get("full")), set.Get("bias1")))
	l2 = tc128.Abs(tc128.Add(tc128.Mul(set.Get("layer2"), l1), set.Get("bias2")))
	l2(func(a *tc128.V) bool {
		o := 0
		for j := 0; j < height; j += blockSize {
			for i := 0; i < width; i += blockSize {
				for y := 0; y < blockSize; y++ {
					for x := 0; x < blockSize; x++ {
						pix := color.RGBA64{
							R: uint16(real(a.X[o])*0xFFFF + .5),
							G: uint16(real(a.X[o+1])*0xFFFF + .5),
							B: uint16(real(a.X[o+2])*0xFFFF + .5),
							A: 0xFFFF,
						}
						coded.SetRGBA64(i+x, j+y, pix)
						o += 3
					}
				}
			}
		}
		return true
	})

	file, err = os.Create(name + "_coded.png")
	if err != nil {
		panic(err)
	}

	err = png.Encode(file, coded)
	if err != nil {
		panic(err)
	}
	file.Close()

	fmt.Println((hiddens*netWidth + netWidth + size*hiddens) * 16)
}

func kroneckerCompress() {
	rand.Seed(7)

	file, err := os.Open(testImage)
	if err != nil {
		panic(err)
	}

	info, err := file.Stat()
	if err != nil {
		panic(err)
	}
	name := info.Name()
	name = name[:strings.Index(name, ".")]

	input, _, err := image.Decode(file)
	if err != nil {
		panic(err)
	}
	file.Close()

	width, height := input.Bounds().Max.X, input.Bounds().Max.Y
	width, height = width/scale, height/scale
	input = resize.Resize(uint(width), uint(height), input, resize.NearestNeighbor)
	width -= width % 1024
	height -= height % 1024
	bounds := image.Rect(0, 0, width, height)
	g := gift.New(
		gift.Crop(bounds),
	)
	cropped := image.NewRGBA64(bounds)
	g.Draw(cropped, input)
	input = cropped

	fmt.Println(math.Log2(float64(width)), width, math.Log2(float64(height)), height, math.Pow(2, 10))
	size := int(math.Log2(float64(width)))
	fmt.Println(float64(4*size) / (1024.0 * 1024.0))

	type Genome struct {
		Genome  []float32
		Fitness float32
	}
	genome := make([]Genome, 128)
	for i := range genome {
		g := Genome{
			Genome:  make([]float32, 4*size),
			Fitness: 0,
		}
		for j := range g.Genome {
			g.Genome[j] = float32(rand.NormFloat64())
		}
		genome[i] = g
	}

	multiply := func(a []float32, b []float32) []float32 {
		sizeA, sizeB := int(math.Sqrt(float64(len(a)))), int(math.Sqrt(float64(len(b))))
		output := make([]float32, 0, len(a)*len(b))
		for x := 0; x < sizeA; x++ {
			for y := 0; y < sizeB; y++ {
				for i := 0; i < sizeA; i++ {
					for j := 0; j < sizeB; j++ {
						output = append(output, a[x*sizeA+i]*b[y*sizeB+j])
					}
				}
			}
		}
		return output
	}
	chain := func(genome []float32) []float32 {
		value := multiply(genome[0:4], genome[4:8])
		for i := 8; i < size*4; i += 4 {
			value = multiply(genome[i:i+4], value)
		}
		return value
	}
	i := 0
	for {
		done := make(chan bool, 8)
		fitness := func(g *Genome) {
			img := chain(g.Genome)
			sum := float32(0.0)
			for y := 0; y < 1024; y++ {
				for x := 0; x < 1024; x++ {
					colors := input.At(x, y)
					r, g, b, _ := colors.RGBA()
					gray := (float32(r)/0xFFFF + float32(g)/0xFFFF + float32(b)/0xFFFF) / 3
					difference := img[y*1024+x] - float32(gray)
					sum += float32(math.Sqrt(float64(difference * difference)))
				}
			}
			g.Fitness = sum
			done <- true
		}
		for j := range genome {
			go fitness(&genome[j])
		}
		for range genome {
			<-done
		}

		sort.Slice(genome, func(i, j int) bool {
			return genome[i].Fitness < genome[j].Fitness
		})
		fmt.Println(i)
		if i >= 256 {
			break
		}

		for i, g := range genome[:10] {
			fmt.Println(i, g.Fitness/(1024*1024))
		}
		genome = genome[:32]
		for i := range genome {
			// swap
			x, y := rand.Intn(10), rand.Intn(10)
			cpx := make([]float32, len(genome[x].Genome))
			copy(cpx, genome[x].Genome)
			gx := Genome{
				Genome:  cpx,
				Fitness: 0,
			}

			cpy := make([]float32, len(genome[y].Genome))
			copy(cpy, genome[y].Genome)
			gy := Genome{
				Genome:  cpy,
				Fitness: 0,
			}

			a, b := rand.Intn(len(cpx)), rand.Intn(len(cpy))
			cpy[a], cpx[b] = cpx[b], cpy[a]
			genome = append(genome, gx)
			genome = append(genome, gy)

			// mutate
			cp := make([]float32, len(genome[i].Genome))
			copy(cp, genome[i].Genome)
			g := Genome{
				Genome:  cp,
				Fitness: 0,
			}
			cp[rand.Intn(len(cp))] += float32(rand.NormFloat64())
			genome = append(genome, g)
		}
		i++
	}

	img := chain(genome[0].Genome)
	coded := image.NewGray(input.Bounds())
	for j := 0; j < height; j++ {
		for i := 0; i < width; i++ {
			pix := color.Gray{
				Y: uint8(img[j*height+i]*0xFF + .5),
			}
			coded.SetGray(i, j, pix)
		}
	}

	file, err = os.Create("image_coded.png")
	if err != nil {
		panic(err)
	}

	err = png.Encode(file, coded)
	if err != nil {
		panic(err)
	}
	file.Close()
}

func main() {
	flag.Parse()

	if *NeuralCompressMode {
		neuralCompress()
		return
	} else if *KroneckerCompressMode {
		kroneckerCompress()
		return
	}
}
