package Overriding_overloading;

public class Test{
    public static void main(String args[]){
 
        Figure f1,f2;
        Square q = new Rectangle();
        Rectangle r = new Rectangle();
        f1 = new Square();
        f2 = new Rectangle();

        f1.stampa(f2); 
        q.stampa(r); 
        f1.stampa(q);
        q.stampa(f1); 
        q.stampa(q); 
    }
}
