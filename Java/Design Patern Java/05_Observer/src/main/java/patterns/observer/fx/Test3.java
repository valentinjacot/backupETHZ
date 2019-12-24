package patterns.observer.fx;

import javafx.beans.binding.NumberBinding;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleIntegerProperty;

public class Test3 {
 
    public static void main(String[] args) {
        IntegerProperty num1 = new SimpleIntegerProperty(1);
        IntegerProperty num2 = new SimpleIntegerProperty(2);
        NumberBinding sum = num1.add(num2);
		sum.addListener((obs, oldVal, newVal) -> {
			System.out.printf("changed from %d to %d%n", oldVal, newVal);
		});
		
        System.out.println(sum.getValue());
        num1.set(2);
        System.out.println(sum.getValue());
    }
}