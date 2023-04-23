

#include <map>

#ifdef TESTING
#define PRIVATE public
#define PROTECTED public
#else
#define PRIVATE private
#define PROTECTED protected
#endif

// Counts references of void* - non-templated to organize all pointer types in a single static var
class RefCounter {
PRIVATE : 
        static std::map<void*, std::size_t> refCounter;
PROTECTED : 
    template <typename T>
    T* inc(std::size_t s)
    {
        T* arr = new T[s];
        refCounter[arr] = 1;
        return arr;
    }
    void inc(void* x)
    {
        refCounter[x]++;
    }
    template <typename T>
    void dec(T* x)
    {
        refCounter[x] -= 1;
        if (refCounter[x] == 0) {
            delete[] x;
        }
    }
};
std::map<void*, std::size_t> RefCounter::refCounter {};

// Templated wrapper for the RefCounter - handles templated array deletion with reassignment
template <typename T>
class Reference : protected RefCounter {
PRIVATE: 
    T* val;
    std::size_t size;
public:
    Reference(const std::size_t s = 1)
    {
        size = s;
        val = inc<T>(size);
    }

    Reference(Reference& r)
    {
        size = r.size;
        inc(r.val);
        val = r.val;
    }

    Reference operator=(Reference r)
    {
        size = r.size;
        dec(val);
        inc(r.val);
        val = r.val;
        return *this;
    }

    ~Reference()
    {
        dec(val);
    }

    T& operator[](const std::size_t x)
    {
        if (x >= size || x < 0)
            throw std::out_of_range("Index is out of range");
        return val[x];
    }
};