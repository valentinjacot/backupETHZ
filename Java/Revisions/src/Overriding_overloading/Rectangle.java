package Overriding_overloading;

public class Rectangle extends Square{
    public void stampa(Rectangle r){
        System.out.println("Rectangle");
    }
    public void stampa(Square r){
        System.out.println("Particular Rectangle");
    }
    public void stampa(Figure f){
        System.out.println("Particular Figure");
    }
}

