package Overriding_overloading;


class Integer extends Rational {
@Override
void add(Real m) { System.out.println("Integer:add(Real)"); }
void add(Rational m) { System.out.println("Integer:add(Rational)"); }
}
