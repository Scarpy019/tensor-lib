#include <iostream>
#include "../testing/test.h"

//define that disables access protection for unit-testing private/protected variables
#define TESTING
#include "./lib/Tensor.h"
#include "./lib/Matmul.h"

int main(){
    SECTION("Reference counting"){
        TEST("Copy counter"){
            {
                Tensor<5, float> t{{5,6,7,2,3}};
                Tensor<5, float> tCopy{t};
                Tensor<5, float> tCopy2 = t;
            }
            for(std::pair<void*, std::size_t> x: RefCounter::refCounter){
                test::equal(x.second, 0, "Reference not cleared.");
            }
        };
    }
    
    SECTION("Modifications"){
        TEST("Simple modification"){
            Tensor<1, float> tens{{7}};
            tens[{3}]=54;
            test::near(tens[{3}], 54);
        };
        TEST("Simple modification (4-dimensional)"){
            Tensor<4, float> tens{{7,1,2,3}};
            tens[{3,0,1,1}]=0.53;
            test::near(tens[{3,0,1,1}], 0.53);
        };
        TEST("Large-scale modification"){
            std::vector<int> records{};
            Tensor<4, float> tens{{4,1,2,3}};
            for(unsigned int i=0;i<4;i++){
                for(unsigned int i2=0;i2<1;i2++){
                    for(unsigned int i3=0;i3<2;i3++){
                        for(unsigned int i4=0;i4<3;i4++){
                            int x=rand()%18;
                            tens[{i,i2,i3,i4}]=x;
                            records.push_back(x);
                        }
                    }
                }
            }
            auto iter=records.begin();
            int arri=0;
            for(unsigned int i=0;i<4;i++){
                for(unsigned int i2=0;i2<1;i2++){
                    for(unsigned int i3=0;i3<2;i3++){
                        for(unsigned int i4=0;i4<3;i4++){
                            test::near(*iter, tens[{i,i2,i3,i4}]);
                            test::near(tens.values.val[arri], *iter);
                            iter++;
                            arri++;
                        }
                    }
                }
            }
        };
    }
    
    SECTION("Slicing"){
        TEST("Pointer synchronicity"){
            Tensor<2, float> t{{3,3}};
            Tensor<1, float> t2=t.slice(1);
            test::equal(t.values.val, t2.values.val);
        };
        TEST("Simple slice value retainment"){
            Tensor<2, float> t{{3,3}};
            t[{1,0}]=1;
            t[{1,1}]=2;
            t[{1,2}]=3;
            Tensor<1, float> t2=t.slice(1);
            test::near(t2[{0}],1);
            test::near(t2[{1}],2);
            test::near(t2[{2}],3);
        };
    }
    
    SECTION("Dimension swapping"){
        TEST("3 dimension tensor swap"){
            Tensor<3, float> t{{3,3,4}};
            t[{1,2,1}]=1;
            t[{1,2,2}]=2;
            t[{2,2,3}]=3;
            Tensor<3, float> t2=t.swapaxes(2, 1);

            test::near(t2[{1,1,2}],1);
            test::near(t2[{1,2,2}],2);
            test::near(t2[{2,3,2}],3);
        };
        TEST("Tensor swap internal memory share"){
            Tensor<3, float> t{{3,3,4}};
            t[{1,2,1}]=1;
            t[{1,2,2}]=2;
            t[{2,2,3}]=3;
            Tensor<3, float> t2=t.swapaxes(2, 1);
            test::equal(t.values.val,t2.values.val);
        };
    }
    
    SECTION("Element iterators"){
        TEST("Iterator constructor"){
            Tensor<3, int> arr{{2,3,2}};
            for(std::size_t i=0;i<2;i++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i3=0;i3<2;i3++){
                        arr[{i,i2,i3}]=i*6+i2*2+i3;
                    }
                }
            }
            Tensor<3, int>::iterator iter{{1,2,0}, arr};
            test::equal(iter.parent, &arr, "Memory addresses do not match");
            test::equal(iter.iterators[0], 1, "Iterator 0 does not match");
            test::equal(iter.iterators[1], 2, "Iterator 1 does not match");
            test::equal(iter.iterators[2], 0, "Iterator 2 does not match");
        };
        THROW_TEST("Out of bounds 1"){
            Tensor<3, int> arr{{2,3,2}};
            for(std::size_t i=0;i<2;i++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i3=0;i3<2;i3++){
                        arr[{i,i2,i3}]=i*6+i2*2+i3;
                    }
                }
            }
            Tensor<3, int>::iterator iter{{1,5,0}, arr};
        };
        THROW_TEST("Out of bounds 2"){
            Tensor<3, int> arr{{2,3,2}};
            for(std::size_t i=0;i<2;i++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i3=0;i3<2;i3++){
                        arr[{i,i2,i3}]=i*6+i2*2+i3;
                    }
                }
            }
            Tensor<3, int>::iterator iter{{1,1,432432}, arr};
        };
        TEST("End iterator construction exception"){
            Tensor<3, int> arr{{2,3,2321}};
            for(std::size_t i=0;i<2;i++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i3=0;i3<2321;i3++){
                        arr[{i,i2,i3}]=i*6+i2*2+i3;
                    }
                }
            }
            Tensor<3, int>::iterator iter{{0,0,2321}, arr};
        };
        TEST("Begin iterator"){
            Tensor<3, int> arr{{2,3,2}};
            for(std::size_t i=0;i<2;i++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i3=0;i3<2;i3++){
                        arr[{i,i2,i3}]=i*6+i2*2+i3;
                    }
                }
            }
            Tensor<3, int>::iterator iter = arr.begin();
            for(int i=0;i<3;i++){
                test::equal(iter.iterators[i], 0);
            }
        };
        TEST("End iterator"){
            Tensor<3, int> arr{{2,3,2}};
            for(std::size_t i=0;i<2;i++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i3=0;i3<2;i3++){
                        arr[{i,i2,i3}]=i*6+i2*2+i3;
                    }
                }
            }
            Tensor<3, int>::iterator iter = arr.end();
            test::equal(iter.parent, &arr, "Parent addresses don't match");
            for(int i=0;i<2;i++){
                test::equal(iter.iterators[i], 0, "var is "+std::to_string(iter.iterators[i]));
            }
            test::equal(iter.iterators[2],2, "var is "+std::to_string(iter.iterators[2]));
        };
        TEST("Comparison operators"){
            Tensor<3, int> arr{{2,3,2}};
            auto a = arr.begin();
            auto b = arr.begin();
            test::equal(a,b);
            auto c = arr.end();
            auto d = arr.end();
            test::equal(c,d);
            test::notEqual(a,c);
            a++;
            test::notEqual(a,b);
            b++;
            test::equal(a,b);
            auto e = b;
            test::equal(e,a);
        };
        TEST("Increment/decrement"){
            Tensor<3, int> arr{{2,3,2}};
            auto a = arr.begin();
            auto b = arr.begin();
            test::equal(a,b);
            auto c = a++;
            test::equal(c,b);
            test::notEqual(a,b);
            b++;
            test::equal(a,b);
            c = ++b;
            test::notEqual(c, a++);
            test::equal(a,c);

        };
        TEST("Internal sequencing (Increments)"){
            Tensor<3, int> arr{{2,3,2}};
            auto x=arr.begin();
            size_t x2[13][3]{
                {0,0,0},
                {1,0,0},
                {0,1,0},
                {1,1,0},
                {0,2,0},
                {1,2,0},
                {0,0,1},
                {1,0,1},
                {0,1,1},
                {1,1,1},
                {0,2,1},
                {1,2,1},
                {0,0,2}
            };
            for(std::size_t i=0;i<12;i++){
                test::match(x.iterators,x2[i],sizeof(std::size_t)*3);
                x++;
            }
            test::match(x.iterators,x2[12],sizeof(std::size_t)*3);
            test::equal(x, arr.end());
        };
        TEST("Internal sequencing (Decrements)"){
            Tensor<3, int> arr{{2,3,2}};
            auto x=arr.end();
            size_t x2[13][3]{
                {0,0,0},
                {1,0,0},
                {0,1,0},
                {1,1,0},
                {0,2,0},
                {1,2,0},
                {0,0,1},
                {1,0,1},
                {0,1,1},
                {1,1,1},
                {0,2,1},
                {1,2,1},
                {0,0,2}
            };
            for(std::size_t i=12;i>0;i--){
                test::match(x.iterators,x2[i],sizeof(std::size_t)*3);
                x--;
            }
            test::match(x.iterators,x2[0],sizeof(std::size_t)*3);
            test::equal(x, arr.begin());
        };
        TEST("operator*"){
            Tensor<3, int> arr{{2,3,2}};
            for(std::size_t i3=0;i3<2;i3++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i=0;i<2;i++){
                        arr[{i,i2,i3}]=i+i2*2+i3*6;
                    }
                }
            }
            int i=0;
            for(auto x=arr.begin();x!=arr.end();x++){
                std::string report="";
                report+="Index ";
                report+=std::to_string(i);
                report+=" does not match. *x=";
                report+=std::to_string(*x);
                test::equal(i,*x, report);
                i++;
            }
        };
        THROW_TEST("End iterator op* throw"){
            Tensor<3, int> arr{{2,3,2321}};
            Tensor<3, int>::iterator iter{{0,0,2321}, arr};
            int x = *iter;
        };
        TEST("Range for loop test"){
            Tensor<3, int> arr{{2,3,2}};
            for(std::size_t i3=0;i3<2;i3++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i=0;i<2;i++){
                        arr[{i,i2,i3}]=i3*6+i2*2+i;
                    }
                }
            }
            std::size_t i=0;
            for(std::size_t x:arr){
                test::equal(i, x);
                i++;
            }
        };
    }

    SECTION("Cloning"){
        TEST("Value retain"){
            Tensor<3, int> arr{{2,3,2}};
            for(std::size_t i3=0;i3<2;i3++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i=0;i<2;i++){
                        arr[{i,i2,i3}]=i3*6+i2*2+i;
                    }
                }
            }
            Tensor<3, int> clon = arr.clone();
            std::size_t i=0;
            for(int x:clon){
                test::equal(i,x);
                i++;
            }
        };
        TEST("synchronicity break"){
            Tensor<3, int> arr{{2,3,2}};
            for(std::size_t i3=0;i3<2;i3++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i=0;i<2;i++){
                        arr[{i,i2,i3}]=i3*6+i2*2+i;
                    }
                }
            }
            Tensor<3, int> clon = arr.clone();
            clon[{1,1,0}]=-5324;
            clon[{0,1,0}]=-534;
            std::size_t i=0;
            for(std::size_t x:arr){
                test::equal(i, x);
                i++;
            }
        };
        TEST("Slice clone"){
            Tensor<3, int> arr{{2,3,2}};
            for(std::size_t i3=0;i3<2;i3++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i=0;i<2;i++){
                        arr[{i,i2,i3}]=i3*6+i2*2+i;
                    }
                }
            }
            Tensor<2, int> clon = arr.slice(1, 2).clone();
            std::size_t i=6;
            for(int x:clon){
                test::equal(i,x);
                i++;
            }
        };
        TEST("Slice swap clone"){
            Tensor<3, int> arr{{2,3,2}};
            for(std::size_t i3=0;i3<2;i3++){
                for(std::size_t i2=0;i2<3;i2++){
                    for(std::size_t i=0;i<2;i++){
                        arr[{i,i2,i3}]=i3*6+i2*2+i;
                    }
                }
            }
            Tensor<2, int> clon = arr.swapaxes(0,2).slice(1).clone().swapaxes(0,1);
            std::size_t i=6;
            for(int x:clon){
                test::equal(i,x);
                i++;
            }
        };
    }

    SECTION("foreach"){
        TEST("addition - 2 tensors"){
            Tensor<2, int> arr{{2,2}};
            arr[{0,0}]=1;
            arr[{0,1}]=2;
            arr[{1,0}]=3;
            arr[{1,1}]=4;
            Tensor<2, int> arr2{{2,2}};
            arr2[{0,0}]=5;
            arr2[{0,1}]=6;
            arr2[{1,0}]=7;
            arr2[{1,1}]=8;
            Tensor<2,int>::foreach<2>({arr,arr2}, [](int*(&vals)[2])->void{
                (*vals[0])+=(*vals[1]);
            });
            test::equal(arr[{0,0}],6);
            test::equal(arr[{0,1}],8);
            test::equal(arr[{1,0}],10);
            test::equal(arr[{1,1}],12);
        };

		TEST("complex - 4 tensors"){
			Tensor<2, int> arr{{2,2}};
            arr[{0,0}]=1;
            arr[{0,1}]=2;
            arr[{1,0}]=3;
            arr[{1,1}]=4;
            Tensor<2, int> arr2{{2,2}};
            arr2[{0,0}]=5;
            arr2[{0,1}]=6;
            arr2[{1,0}]=7;
            arr2[{1,1}]=8;
			Tensor<2, int> arr3{{2,2}};
            arr3[{0,0}]=2;
            arr3[{0,1}]=2;
            arr3[{1,0}]=2;
            arr3[{1,1}]=2;
			Tensor<2, int> arr4{{2,2}};
            arr4[{0,0}]=4;
            arr4[{0,1}]=4;
            arr4[{1,0}]=3;
            arr4[{1,1}]=2;
            Tensor<2,int>::foreach<4>({arr,arr2, arr3, arr4}, [](int*(&vals)[4])->void{
                (*vals[0]) += (*vals[1]) * (*vals[2]) + (*vals[3]);
            });
            test::equal(arr[{0,0}],15);
            test::equal(arr[{0,1}],18);
            test::equal(arr[{1,0}],20);
            test::equal(arr[{1,1}],22);
		};
    }

    SECTION("operator +="){
        TEST("operator+="){
            Tensor<2, int> arr{{2,2}};
            arr[{0,0}]=1;
            arr[{0,1}]=2;
            arr[{1,0}]=3;
            arr[{1,1}]=4;
            Tensor<2, int> arr2{{2,2}};
            arr2[{0,0}]=5;
            arr2[{0,1}]=6;
            arr2[{1,0}]=7;
            arr2[{1,1}]=8;
            arr+=arr2;
            test::equal(arr[{0,0}],6);
            test::equal(arr[{0,1}],8);
            test::equal(arr[{1,0}],10);
            test::equal(arr[{1,1}],12);
        };
        TEST("chaining +="){
            Tensor<2, int> arr{{2,2}};
            arr[{0,0}]=1;
            arr[{0,1}]=2;
            arr[{1,0}]=3;
            arr[{1,1}]=4;
            Tensor<2, int> arr2{{2,2}};
            arr2[{0,0}]=5;
            arr2[{0,1}]=6;
            arr2[{1,0}]=7;
            arr2[{1,1}]=8;
            (arr+=arr2)+=arr;
            test::equal(arr[{0,0}],12);
            test::equal(arr[{0,1}],16);
            test::equal(arr[{1,0}],20);
            test::equal(arr[{1,1}],24);
        };
        TEST("broadcasting"){
            Tensor<2, int> arr{{2,2}};
            arr[{0,0}]=1;
            arr[{0,1}]=2;
            arr[{1,0}]=3;
            arr[{1,1}]=4;
            Tensor<1, int> arr2{{2}};
            arr2[{0}]=5;
            arr2[{1}]=6;
            arr+=arr2;
            test::equal(arr[{0,0}],6);
            test::equal(arr[{0,1}],8);
            test::equal(arr[{1,0}],8);
            test::equal(arr[{1,1}],10);
        };
        TEST("broadcasting 2"){
            Tensor<3, int> arr{{2,2,2}};
            arr[{0,0,0}]=1;
            arr[{0,0,1}]=2;
            arr[{0,1,0}]=3;
            arr[{0,1,1}]=4; 
            arr[{1,0,0}]=5;
            arr[{1,0,1}]=6;
            arr[{1,1,0}]=7;
            arr[{1,1,1}]=8;
            Tensor<1, int> arr2{{2}};
            arr2[{0}]=5;
            arr2[{1}]=6;
            arr+=arr2;
            test::equal(arr[{0,0,0}],6);
            test::equal(arr[{0,0,1}],8);
            test::equal(arr[{0,1,0}],8);
            test::equal(arr[{0,1,1}],10);
            test::equal(arr[{1,0,0}],10);
            test::equal(arr[{1,0,1}],12);
            test::equal(arr[{1,1,0}],12);
            test::equal(arr[{1,1,1}],14);
        };
    }

	SECTION("operator -="){
        TEST("operator-="){
            Tensor<2, int> arr{{2,2}};
            arr[{0,0}]=1;
            arr[{0,1}]=2;
            arr[{1,0}]=3;
            arr[{1,1}]=4;
            Tensor<2, int> arr2{{2,2}};
            arr2[{0,0}]=5;
            arr2[{0,1}]=6;
            arr2[{1,0}]=7;
            arr2[{1,1}]=8;
            arr-=arr2;
            test::equal(arr[{0,0}],-4);
            test::equal(arr[{0,1}],-4);
            test::equal(arr[{1,0}],-4);
            test::equal(arr[{1,1}],-4);
        };
        TEST("chaining -="){
            Tensor<2, int> arr{{2,2}};
            arr[{0,0}]=1;
            arr[{0,1}]=2;
            arr[{1,0}]=3;
            arr[{1,1}]=4;
            Tensor<2, int> arr2{{2,2}};
            arr2[{0,0}]=5;
            arr2[{0,1}]=6;
            arr2[{1,0}]=7;
            arr2[{1,1}]=8;
            (arr-=arr2)-=arr;
            test::equal(arr[{0,0}],0);
            test::equal(arr[{0,1}],0);
            test::equal(arr[{1,0}],0);
            test::equal(arr[{1,1}],0);
        };
        TEST("broadcasting"){
            Tensor<2, int> arr{{2,2}};
            arr[{0,0}]=1;
            arr[{0,1}]=2;
            arr[{1,0}]=3;
            arr[{1,1}]=4;
            Tensor<1, int> arr2{{2}};
            arr2[{0}]=5;
            arr2[{1}]=6;
            arr-=arr2;
            test::equal(arr[{0,0}],-4);
            test::equal(arr[{0,1}],-4);
            test::equal(arr[{1,0}],-2);
            test::equal(arr[{1,1}],-2);
        };
        TEST("broadcasting 2"){
            Tensor<3, int> arr{{2,2,2}};
            arr[{0,0,0}]=1;
            arr[{0,0,1}]=2;
            arr[{0,1,0}]=3;
            arr[{0,1,1}]=4; 
            arr[{1,0,0}]=5;
            arr[{1,0,1}]=6;
            arr[{1,1,0}]=7;
            arr[{1,1,1}]=8;
            Tensor<1, int> arr2{{2}};
            arr2[{0}]=5;
            arr2[{1}]=6;
            arr-=arr2;
            test::equal(arr[{0,0,0}],-4);
            test::equal(arr[{0,0,1}],-4);
            test::equal(arr[{0,1,0}],-2);
            test::equal(arr[{0,1,1}],-2);
            test::equal(arr[{1,0,0}],0);
            test::equal(arr[{1,0,1}],0);
            test::equal(arr[{1,1,0}],2);
            test::equal(arr[{1,1,1}],2);
        };
    }
	
	SECTION("Expand"){
		TEST("expansion 1"){
			Tensor<2, int> arr{{2,3}};
            arr[{0,0}]=1;
            arr[{0,1}]=2;
			arr[{0,2}]=5;
            arr[{1,0}]=3;
            arr[{1,1}]=4;
			arr[{1,2}]=6;
			Tensor arr2 = arr.expand();
			test::equal(arr2.dimensions[0], 1);
			test::equal(arr2.dimensions[1], 2);
			test::equal(arr2.dimensions[2], 3);
			test::equal(arr2.dimensionIncrementors[0], 1);
			test::equal(arr2.dimensionIncrementors[1], arr.dimensionIncrementors[0]);
			test::equal(arr2.dimensionIncrementors[2], arr.dimensionIncrementors[1]);
		};
		TEST("expansion 2"){
			Tensor<2, int> arr{{2,3}};
            arr[{0,0}]=1;
            arr[{0,1}]=2;
			arr[{0,2}]=5;
            arr[{1,0}]=3;
            arr[{1,1}]=4;
			arr[{1,2}]=6;
			Tensor arr2 = arr.expand(1);
			test::equal(arr2.dimensions[0], 2);
			test::equal(arr2.dimensions[1], 1);
			test::equal(arr2.dimensions[2], 3);
			test::equal(arr2.dimensionIncrementors[0], arr.dimensionIncrementors[0]);
			test::equal(arr2.dimensionIncrementors[1], 1);
			test::equal(arr2.dimensionIncrementors[2], arr.dimensionIncrementors[1]);
		};
		TEST("expansion 3"){
			Tensor<1, int> arr{{2}};
            arr[{0}]=1;
            arr[{1}]=2;
			Tensor arr2 = arr.expand();
			test::equal(arr2.dimensions[1], 2);
			test::equal(arr2.dimensions[0], 1);
			test::equal(arr2.dimensionIncrementors[1], arr.dimensionIncrementors[0]);
			test::equal(arr2.dimensionIncrementors[0], 1);
		};
	}
	
	SECTION("Matmul out variant"){
		TEST("regular matmul"){
			Tensor<2, int> x1{{2,2}};
			x1[{0,0}] = 4;
			x1[{0,1}] = 1;
			x1[{1,0}] = -6;
			x1[{1,1}] = 8;
			Tensor<2, int> x2{{2,2}};
			x2[{0,0}] = 4;
			x2[{0,1}] = -18;
			x2[{1,0}] = 2;
			x2[{1,1}] = -3;
			Tensor<2, int> x3{{2,2}};
			matmul(x1,x2,x3);
			Tensor<2, int> expected{{2,2}};
			expected[{0,0}] = 18;
			expected[{0,1}] = -75;
			expected[{1,0}] = -8;
			expected[{1,1}] = 84;
			test::equal(x3[{0,0}], expected[{0,0}]);
			test::equal(x3[{0,1}], expected[{0,1}]);
			test::equal(x3[{1,0}], expected[{1,0}]);
			test::equal(x3[{1,1}], expected[{1,1}]);
		};
		TEST("broadcast to x2"){
			Tensor<2, int> x1{{2,2}};
			x1[{0,0}] = 4;
			x1[{0,1}] = 1;
			x1[{1,0}] = -6;
			x1[{1,1}] = 8;
			Tensor<3, int> x2{{2,2,2}};
			x2[{0,0,0}] = 4;
			x2[{0,0,1}] = -18;
			x2[{0,1,0}] = 2;
			x2[{0,1,1}] = -3;
			x2[{1,0,0}] = 9;
			x2[{1,0,1}] = -1;
			x2[{1,1,0}] = -92;
			x2[{1,1,1}] = 3;
			Tensor<3, int> x3{{2,2,2}};
			matmul(x1,x2,x3);
			Tensor<3, int> expected{{2,2,2}};
			expected[{0,0,0}] = 18;
			expected[{0,0,1}] = -75;
			expected[{0,1,0}] = -8;
			expected[{0,1,1}] = 84;
			test::equal(x3[{0,0,0}], expected[{0,0,0}]);
			test::equal(x3[{0,0,1}], expected[{0,0,1}]);
			test::equal(x3[{0,1,0}], expected[{0,1,0}]);
			test::equal(x3[{0,1,1}], expected[{0,1,1}]);
			expected[{1,0,0}] = -56;
			expected[{1,0,1}] = -1;
			expected[{1,1,0}] = -790;
			expected[{1,1,1}] = 30;
			test::equal(x3[{1,0,0}], expected[{1,0,0}]);
			test::equal(x3[{1,0,1}], expected[{1,0,1}]);
			test::equal(x3[{1,1,0}], expected[{1,1,0}]);
			test::equal(x3[{1,1,1}], expected[{1,1,1}]);
		};
		TEST("broadcast to x1"){
			Tensor<2, int> x1{{2,2}};
			x1[{0,0}] = 4;
			x1[{0,1}] = 1;
			x1[{1,0}] = -6;
			x1[{1,1}] = 8;
			Tensor<3, int> x2{{2,2,2}};
			x2[{0,0,0}] = 4;
			x2[{0,0,1}] = -18;
			x2[{0,1,0}] = 2;
			x2[{0,1,1}] = -3;
			x2[{1,0,0}] = 9;
			x2[{1,0,1}] = -1;
			x2[{1,1,0}] = -92;
			x2[{1,1,1}] = 3;
			Tensor<3, int> x3{{2,2,2}};
			matmul(x2,x1,x3);
			Tensor<3, int> expected{{2,2,2}};
			expected[{0,0,0}] = 124;
			expected[{0,0,1}] = -140;
			expected[{0,1,0}] = 26;
			expected[{0,1,1}] = -22;
			test::equal(x3[{0,0,0}], expected[{0,0,0}]);
			test::equal(x3[{0,0,1}], expected[{0,0,1}]);
			test::equal(x3[{0,1,0}], expected[{0,1,0}]);
			test::equal(x3[{0,1,1}], expected[{0,1,1}]);
			expected[{1,0,0}] = 42;
			expected[{1,0,1}] = 1;
			expected[{1,1,0}] = -386;
			expected[{1,1,1}] = -68;
			test::equal(x3[{1,0,0}], expected[{1,0,0}]);
			test::equal(x3[{1,0,1}], expected[{1,0,1}]);
			test::equal(x3[{1,1,0}], expected[{1,1,0}]);
			test::equal(x3[{1,1,1}], expected[{1,1,1}]);
		};
		TEST("3-dimensional matmul"){
			Tensor<3, int> x1{{2,2,2}};
			x1[{0,0,0}] = 4;
			x1[{0,0,1}] = 1;
			x1[{0,1,0}] = -6;
			x1[{0,1,1}] = 8;
			x1[{1,0,0}] = 9;
			x1[{1,0,1}] = -1;
			x1[{1,1,0}] = -92;
			x1[{1,1,1}] = 3;
			Tensor<3, int> x2{{2,2,2}};
			x2[{0,0,0}] = 4;
			x2[{0,0,1}] = -18;
			x2[{0,1,0}] = 2;
			x2[{0,1,1}] = -3;
			x2[{1,0,0}] = 9;
			x2[{1,0,1}] = -1;
			x2[{1,1,0}] = -92;
			x2[{1,1,1}] = 3;
			Tensor<3, int> x3{{2,2,2}};
			matmul(x1,x2,x3);
			Tensor<3, int> expected{{2,2,2}};
			expected[{0,0,0}] = 18;
			expected[{0,0,1}] = -75;
			expected[{0,1,0}] = -8;
			expected[{0,1,1}] = 84;
			test::equal(x3[{0,0,0}], expected[{0,0,0}]);
			test::equal(x3[{0,0,1}], expected[{0,0,1}]);
			test::equal(x3[{0,1,0}], expected[{0,1,0}]);
			test::equal(x3[{0,1,1}], expected[{0,1,1}]);
			expected[{1,0,0}] = 173;
			expected[{1,0,1}] = -12;
			expected[{1,1,0}] = -1104;
			expected[{1,1,1}] = 101;
			test::equal(x3[{1,0,0}], expected[{1,0,0}]);
			test::equal(x3[{1,0,1}], expected[{1,0,1}]);
			test::equal(x3[{1,1,0}], expected[{1,1,0}]);
			test::equal(x3[{1,1,1}], expected[{1,1,1}]);
		};
		TEST("2x2 @ 2"){
			Tensor<2, int> x1{{2,2}};
			x1[{0,0}] = 4;
			x1[{0,1}] = 1;
			x1[{1,0}] = -6;
			x1[{1,1}] = 8;
			Tensor<2, int> x2{{2,1}};
			x2[{0,0}] = 4;
			x2[{1,0}] = -18;
			Tensor<2, int> x3{{2,1}};
			matmul(x1,x2,x3);
			Tensor<2, int> expected{{2,1}};
			expected[{0,0}] = -2;
			expected[{1,0}] = -168;
			test::equal(x3[{0}], expected[{0}]);
			test::equal(x3[{1}], expected[{1}]);
		};
	}
    test::start();
}

