package patterns.observer.fx;

import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;

public class Test1 {
	public static void main(String[] args) {
		StringProperty p1 = new SimpleStringProperty("p1");
		IntegerProperty p2 = new SimpleIntegerProperty();
		p2.bind(p1.length());
		
		p2.addListener(o -> {System.out.println("value was invalidated");});
		p2.addListener((observable, oldValue, newValue) -> {
			System.out.printf("value changed from %s to %s\n", oldValue, newValue);
		});
		
		p1.setValue("p2");
		p1.setValue("p3");
		p1.setValue("p4");
		
		System.out.println(p2.getValue());
		
		p1.setValue("p5");
		p1.setValue("xxx");
		p1.setValue("p5");
		
//		p2.setValue(3);

	}
}