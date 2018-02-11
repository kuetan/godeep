package read

import (
	"path"
	"compress/gzip"
	"encoding/binary"
	"io"
	"os"
	"fmt"
)

type RawImage []byte

type Set struct {
	NRow   int
	NCol   int
	Images []RawImage
	Labels []Label
}

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	Width      = 28
	Height     = 28
)

func ReadSet(iname, lname string) (set *Set, err error) {
	set = &Set{}
	if set.NRow, set.NCol, set.Images, err = ReadImageFile(iname); err != nil {
		return nil, err
	}
	if set.Labels, err = ReadLabelFile(lname); err != nil {
		return nil, err
	}
	return
}

func ReadImageFile(name string) (rows, cols int, imgs []RawImage, err error) {
	f, err := os.Open(name)
	if err != nil {
		return 0, 0, nil, err
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		return 0, 0, nil, err
	}
	return readImageFile(z)
}

func readImageFile(r io.Reader) (rows, cols int, imgs []RawImage, err error) {
	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return 0, 0, nil, err
	}
	if magic != imageMagic {
		return 0, 0, nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return 0, 0, nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return 0, 0, nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return 0, 0, nil, err
	}
	imgs = make([]RawImage, n)
	m := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgs[i] = make(RawImage, m)
		m_, err := io.ReadFull(r, imgs[i])
		if err != nil {
			return 0, 0, nil, err
		}
		if m_ != int(m) {
			return 0, 0, nil, os.ErrInvalid
		}
	}
	return int(nrow), int(ncol), imgs, nil
}

// Label is a digit label in 0 to 9
type Label uint8

// ReadLabelFile opens the named label file (training or test), parses it and
// returns all labels in order.
func ReadLabelFile(name string) (labels []Label, err error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	return readLabelFile(z)
}

func readLabelFile(r io.Reader) (labels []Label, err error) {
	var (
		magic int32
		n     int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels = make([]Label, n)
	for i := 0; i < int(n); i++ {
		var l Label
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil, err
		}
		labels[i] = l
	}
	return labels, nil
}


func Load(dir string) (train, test *Set, err error) {
	if train, err = ReadSet(path.Join(dir, "train-images-idx3-ubyte.gz"), path.Join(dir, "train-labels-idx1-ubyte.gz")); err != nil {
		return nil, nil, err
	}
	if test, err = ReadSet(path.Join(dir, "t10k-images-idx3-ubyte.gz"), path.Join(dir, "t10k-labels-idx1-ubyte.gz")); err != nil {
		return nil, nil, err
	}
	return
}

func main(){
	//train, test, err := Load("./dataset/mnist/")
	//train = train
	//test = test
	//if err == nil {
	// 	fmt.Println(test.Images[0])
	//}
	MNISTLoader.LoadTrain()
}
