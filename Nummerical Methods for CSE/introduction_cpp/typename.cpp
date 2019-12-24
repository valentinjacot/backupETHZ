// typename
//____________________________

template <class T>
void template_function(){
	T::t * x; // c++ this is a multiplication and not a pointer NO SENSE
	
}


template <class T>
void template_function(){
	typename T::t * x; // now ok (pointer)
	
}


// other expl


template <typename T>
struct tStruct {
	template <typename U>
	U tMemberFunc(void) {return U();}
};
