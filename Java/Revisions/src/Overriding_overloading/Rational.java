package Overriding_overloading;

class Rational extends Real {
void add(Real f) { System.out.println("Rational:add(Real)"); }
@Override
void add(Integer s) { System.out.println("Rational:add(Integer)"); }
}