template <class T>
T const& min(T const& x, T const& y)
{
return x < y ? x : y;
}
template <class R, class U, class T> // min<double>(1,3.141); specify the output type
R const& min(U const& x, T const& y)
{
return (x < y ? static_cast<R>(x) : static_cast<R>(y));
}
