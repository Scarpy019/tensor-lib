#include <stdexcept>
#include <array>
#include "References.h"

#ifdef TESTING
#define PRIVATE public
#define PROTECTED public
#else
#define PRIVATE private
#define PROTECTED protected
#endif



template <int DIMENSION_COUNT, typename T>
class Tensor {
PRIVATE: 

    std::size_t dimensions[DIMENSION_COUNT];
    std::size_t dimensionIncrementors[DIMENSION_COUNT];
    std::size_t offset = 0;
    Reference<T> values;

public:
    // initializes a Tensor with set dimensions
    Tensor(const std::size_t (&list)[DIMENSION_COUNT]);

    Tensor();

    // copy constructor
    Tensor(Tensor& t);

    Tensor operator=(Tensor t);

    T& operator[](const std::size_t (&list)[DIMENSION_COUNT]);

    Tensor& operator+=(Tensor t);
    
    template<int O_DIM>
    Tensor& operator+=(Tensor<O_DIM, T> t);

    class iterator{
    PRIVATE:
        Tensor* parent=nullptr;
        std::size_t iterators[DIMENSION_COUNT];
    public:
        iterator();
        iterator(const std::size_t (&startIterators)[DIMENSION_COUNT], Tensor &parentTensor);
        iterator(iterator& it);

        // assignment operator

        iterator& operator=(iterator iter);

        // dereference operators

        T& operator*();

        // increment/decrement operators
        iterator operator++();
        iterator operator++(int);

        iterator operator--();
        iterator operator--(int);

        // comparison operators

        bool operator==(iterator t);
        bool operator!=(iterator t);
    };

    //returns the begin iterator of the tensor
    iterator begin();

    //returns the end iterator of the tensor
    iterator end();

    // acquires a slice of the tensor along a dimension
    Tensor<DIMENSION_COUNT-1, T> slice(std::size_t x, std::size_t dim=0);

    // swaps 2 axes dim1 and dim2, returns new tensor with swapped dimensions but shared memory
    Tensor swapaxes(const std::size_t dim1, const std::size_t dim2);

    //creates a seperate memory tensor and copies over all items
    Tensor clone();

    template<int T_COUNT>
    static void foreach( std::array<Tensor, T_COUNT> tensors, //Tensor* (tensors)[T_COUNT], /*std::initializer_list<Tensor<DIMENSION_COUNT, T>>,*/
		void(*func)(T*(&values)[T_COUNT])){
			Tensor ts[T_COUNT];
			//auto iter=tens.begin();
			for(int i=0;i<T_COUNT;i++){
				ts[i]=tensors[i];
				//iter++;
			}
			static_assert(T_COUNT>0, "Tensor count has to be greater than 0");
			//check if all sizes match
			for(std::size_t x=0;x<DIMENSION_COUNT;x++){
				std::size_t zeroSize=ts[0].dimensions[x];
				for(std::size_t t=0;t<T_COUNT;t++){
					if(zeroSize!=ts[t].dimensions[x])throw std::invalid_argument("Dimensions don't match");
				}
			}
			T* vals[T_COUNT];
			Tensor::iterator its[T_COUNT];
			for(int i=0;i<T_COUNT;i++){
				its[i]=ts[i].begin();
			}
			while(its[0]!=ts[0].end()){
				for(int i=0;i<T_COUNT;i++){
					vals[i]=&*its[i];
					
				}
				func(vals);
				for(int i=0;i<T_COUNT;i++){
					its[i]++;
				}
			}
		};

};

// ------------------------------------------Constructors----------------------------------------

template <int DIMENSION_COUNT, typename T>
Tensor<DIMENSION_COUNT, T>::Tensor(const std::size_t (&list)[DIMENSION_COUNT])
{
    static_assert(DIMENSION_COUNT!=0, "Only Non-zero dimensional tensors are supported at this time.");
    std::size_t totalSize = 1;
    // Set up incrementors, dimensions
    for (int i = 0; i < DIMENSION_COUNT; i++) {
        dimensions[i] = list[i];
        totalSize *= dimensions[i];
        dimensionIncrementors[i] = 1;
    }

    for (int i = 1; i < DIMENSION_COUNT; i++) {
        for (int i2 = 0; i2 < i; i2++) {
            dimensionIncrementors[i2] *= dimensions[i];
        }
    }

    values = Reference<T> { totalSize };
}

template <int DIMENSION_COUNT, typename T>
Tensor<DIMENSION_COUNT, T>::Tensor(){
    static_assert(DIMENSION_COUNT!=0, "Only Non-zero dimensional tensors are supported at this time.");
}

template <int DIMENSION_COUNT, typename T>
Tensor<DIMENSION_COUNT, T>::Tensor(Tensor& t)
{
    static_assert(DIMENSION_COUNT!=0, "Only Non-zero dimensional tensors are supported at this time.");
    for (std::size_t i = 0; i < DIMENSION_COUNT; i++) {
        dimensions[i] = t.dimensions[i];
        dimensionIncrementors[i] = t.dimensionIncrementors[i];
    }
    offset = t.offset;
    values = t.values;
}

// --------------------------------------operators------------------------------------------------------

template <int DIMENSION_COUNT, typename T>
Tensor<DIMENSION_COUNT, T> Tensor<DIMENSION_COUNT, T>::operator=(Tensor<DIMENSION_COUNT, T> t)
{

    for (std::size_t i = 0; i < DIMENSION_COUNT; i++) {
        dimensions[i] = t.dimensions[i];
        dimensionIncrementors[i] = t.dimensionIncrementors[i];
    }
    offset = t.offset;
    values = t.values;
    return *this;
}

template <int DIMENSION_COUNT, typename T>
T& Tensor<DIMENSION_COUNT, T>::operator[](const std::size_t (&list)[DIMENSION_COUNT])
{
    std::size_t index = offset;
    for (std::size_t  i = 0; i < DIMENSION_COUNT; i++) {
        if (list[i] >= dimensions[i]) {
            std::string errmsg = "Index ";
            errmsg += std::to_string(i);
            errmsg += " is out of range";
            throw std::out_of_range(errmsg);
        }
        index += list[i] * dimensionIncrementors[i];
    }
    return values[index];
}

template <int DIMENSION_COUNT, typename T>
Tensor<DIMENSION_COUNT, T>& Tensor<DIMENSION_COUNT, T>::operator+=(Tensor t){

    //check dimensions
    for(int i=0;i<DIMENSION_COUNT;i++){
        if(dimensions[i]!=t.dimensions[i])throw std::invalid_argument("Tensor dimensions don't match");
    }
    
    auto tIter=t.begin();
    for(T& x:*this){
        x+=*tIter;
        tIter++;
    }

    return *this;
} 

template <int DIMENSION_COUNT, typename T>
template <int O_DIM>
Tensor<DIMENSION_COUNT, T>& Tensor<DIMENSION_COUNT, T>::operator+=(Tensor<O_DIM, T> t)
{

    //check dimensions
    static_assert(DIMENSION_COUNT>O_DIM, "Dimension counts unfit for broadcasting");
    
    const int DIM_DIFF = DIMENSION_COUNT - O_DIM;

    for(std::size_t i=0;i<dimensions[0];i++){
        this->slice(i)+=t;
    }
    
    
    return *this;
}



// ------------------------------------iterator methods------------------------------------


template <int DIMENSION_COUNT, typename T>
typename Tensor<DIMENSION_COUNT, T>::iterator Tensor<DIMENSION_COUNT, T>::begin()
{
    std::size_t it[DIMENSION_COUNT];
    for(int i=0;i<DIMENSION_COUNT;i++){
        it[i]=0;
    }
    return iterator(it, *this);
}

template <int DIMENSION_COUNT, typename T>
typename Tensor<DIMENSION_COUNT, T>::iterator Tensor<DIMENSION_COUNT, T>::end()
{
    std::size_t it[DIMENSION_COUNT];
    for(int i=0;i<DIMENSION_COUNT-1;i++){
        it[i]=0;
    }
    it[DIMENSION_COUNT-1]=dimensions[DIMENSION_COUNT-1];
    return iterator(it, *this);
}

// ----------------------------------Tensor manipulators-------------------------------------

template <int DIMENSION_COUNT, typename T>
Tensor<DIMENSION_COUNT-1, T> Tensor<DIMENSION_COUNT, T>::slice(std::size_t x, std::size_t dim)
{
    static_assert(DIMENSION_COUNT!=1, "Unable to slice one dimensional tensors");
    Tensor<DIMENSION_COUNT-1, T> s{};
    s.values=values;
    s.offset=offset+dimensionIncrementors[dim]*x;
    std::size_t ni=0;
    for(std::size_t i=0;i<DIMENSION_COUNT;i++){
        if(i==dim)continue;
        s.dimensions[ni]=dimensions[i];
        s.dimensionIncrementors[ni]=dimensionIncrementors[i];
        ni++;
    }
    return s;
}

template <int DIMENSION_COUNT, typename T>
Tensor<DIMENSION_COUNT, T> Tensor<DIMENSION_COUNT, T>::swapaxes(const std::size_t dim1, const std::size_t dim2)
{
    if(dim1 >= DIMENSION_COUNT) throw std::out_of_range("dim1 out of range");
    if(dim2 >= DIMENSION_COUNT) throw std::out_of_range("dim2 out of range");
    Tensor cpy{*this};
    
    // copy incrementors
    std::size_t temp = cpy.dimensionIncrementors[dim1];
    cpy.dimensionIncrementors[dim1] = cpy.dimensionIncrementors[dim2];
    cpy.dimensionIncrementors[dim2] = temp;

    //copy dimensions
    temp = cpy.dimensions[dim1];
    cpy.dimensions[dim1] = cpy.dimensions[dim2];
    cpy.dimensions[dim2] = temp;
    
    return cpy;
}

template <int DIMENSION_COUNT, typename T>
Tensor<DIMENSION_COUNT, T> Tensor<DIMENSION_COUNT, T>::clone()
{
    Tensor cpy{dimensions};
    iterator newIter=cpy.begin();
    for(T& val: *this){
        *newIter = val;
        newIter++;
    }
    return cpy;
}



// --------------------------Tensor iterator-----------------------------------
// ----------------------------------------------------------------------------

// ----------------constructors----------------------

template <int DIMENSION_COUNT, typename T>
Tensor<DIMENSION_COUNT, T>::iterator::iterator()
{

}

template <int DIMENSION_COUNT, typename T>
Tensor<DIMENSION_COUNT, T>::iterator::iterator(const std::size_t (&startIterators)[DIMENSION_COUNT], Tensor<DIMENSION_COUNT, T> &parentTensor): parent(&parentTensor)
{
    static_assert(DIMENSION_COUNT>0, "0-dimensional tensors are not supported.");
    //iterator bounds check (end iterator needs an extra if statement)
    if(startIterators[DIMENSION_COUNT-1]==parent->dimensions[DIMENSION_COUNT-1]){
        for(std::size_t i=0;i<DIMENSION_COUNT-1;i++){
            if(startIterators[i]!=0)std::out_of_range("Starting iterators are out of range");
        }
    }else{
        for(std::size_t i=0;i<DIMENSION_COUNT; i++){
            if(parent->dimensions[i]<=startIterators[i])throw std::out_of_range("Starting iterators are out of range");
        }
    }
    //copy over values
    for(std::size_t i=0;i<DIMENSION_COUNT;i++){
        iterators[i]=startIterators[i];
    }
}

template <int DIMENSION_COUNT, typename T>
Tensor<DIMENSION_COUNT, T>::iterator::iterator(iterator& it): parent(it.parent)
{
    for(int i=0;i<DIMENSION_COUNT;i++){
        iterators[i]=it.iterators[i];
    }
}

// -----------------------------------------operators---------------------------------------

template <int DIMENSION_COUNT, typename T>
typename Tensor<DIMENSION_COUNT, T>::iterator& Tensor<DIMENSION_COUNT, T>::iterator::operator=(iterator iter)
{
    parent=iter.parent;
    for(int i=0;i<DIMENSION_COUNT;i++){
        iterators[i]=iter.iterators[i];
    }
    return *this;
}

template <int DIMENSION_COUNT, typename T>
T& Tensor<DIMENSION_COUNT, T>::iterator::operator*()
{
    for(int i=0;i<DIMENSION_COUNT;i++){
        if(iterators[i]>=parent->dimensions[i])throw std::out_of_range("Iterator is attempting to access an invalid location");
    }
    return (*parent)[iterators];
}

template <int DIMENSION_COUNT, typename T>
typename Tensor<DIMENSION_COUNT, T>::iterator Tensor<DIMENSION_COUNT, T>::iterator::operator++(){
    iterators[0]++;

    for(int i=0;i<DIMENSION_COUNT-1;i++){
        if(iterators[i]==parent->dimensions[i]){
            iterators[i]=0;
            iterators[i+1]++;
        }else{
            break;
        }
    }
    //preventative measure against overflow
    if(iterators[DIMENSION_COUNT-1]==parent->dimensions[DIMENSION_COUNT-1]){
        //check all other iterators if end hasn't been overstepped
        for(int i=0;i<DIMENSION_COUNT-1;i++){
            if(iterators[i]!=0)throw std::out_of_range("Iterator out of range");
        }
    }
    return *this;
}

template <int DIMENSION_COUNT, typename T>
typename Tensor<DIMENSION_COUNT, T>::iterator Tensor<DIMENSION_COUNT, T>::iterator::operator++(int){
    iterator copy{*this};
    ++(*this);
    return copy;
}

template <int DIMENSION_COUNT, typename T>
typename Tensor<DIMENSION_COUNT, T>::iterator Tensor<DIMENSION_COUNT, T>::iterator::operator--(){
    bool carry = true;
    for(int i=0; i<DIMENSION_COUNT;i++){
        if(iterators[i]!=0){
            carry=false;
            iterators[i]--;
            break;
        }else{
            iterators[i]=parent->dimensions[i]-1;
        }
    }
    if(carry){
        throw std::out_of_range("Iterator gone out of range");
    }
    return *this;
}

template <int DIMENSION_COUNT, typename T>
typename Tensor<DIMENSION_COUNT, T>::iterator Tensor<DIMENSION_COUNT, T>::iterator::operator--(int){
    iterator copy{*this};
    --(*this);
    return copy;
}


template <int DIMENSION_COUNT, typename T>
bool Tensor<DIMENSION_COUNT, T>::iterator::operator==(iterator t){
    bool eq=true;
    // same origin
    eq = eq && t.parent->values.val==parent->values.val;
    // same offset
    eq = eq && t.parent->offset==parent->offset;
    //same dimensions, incrementors and iterators
    for(int i=0;i<DIMENSION_COUNT;i++){
        eq = eq && parent->dimensions[i]==t.parent->dimensions[i];
        eq = eq && parent->dimensionIncrementors[i]==t.parent->dimensionIncrementors[i];
        eq = eq && iterators[i]==t.iterators[i];
    }
    return eq;
}

template <int DIMENSION_COUNT, typename T>
bool Tensor<DIMENSION_COUNT, T>::iterator::operator!=(iterator t){
    return !(*this==t);
}
