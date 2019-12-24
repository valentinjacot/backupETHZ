package patterns.observer.fx;

import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;

public class Test2 {
    public static void main(String[] args) {
        StringProperty prop1 = new SimpleStringProperty("");
        StringProperty prop2 = new SimpleStringProperty("");

        prop2.bindBidirectional(prop1);

        System.out.println("prop1.isBound() = " + prop1.isBound());
        System.out.println("prop2.isBound() = " + prop2.isBound());

        prop1.set("value1");
        System.out.println(prop1.get());
        System.out.println(prop2.get());

        prop2.set("value2");
        System.out.println(prop1.get());
        System.out.println(prop2.get());
    }
}