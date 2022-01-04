// Copyright 2022 The Hyperion Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
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

func main() {
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
