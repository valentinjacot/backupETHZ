package Overriding_overloading;
class Main{
public static void main(String[] args) {
Real r = new Integer();
Rational q = new Integer();
Integer i = new Integer();
// a)
r.add(i); //
//
// b)
q.add(q); //
//
// c)
i.add(i); //
//
// d)
i.add(q); //
//
}
}
