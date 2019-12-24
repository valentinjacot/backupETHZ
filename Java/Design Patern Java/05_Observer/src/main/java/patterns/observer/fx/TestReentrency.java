package patterns.observer.fx;

import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;

public class TestReentrency {

	public static void main(String[] args) {
		IntegerProperty i1 = new SimpleIntegerProperty(0);
		i1.addListener(new Listener("Listener1: "));
		i1.addListener((v, oldVal, newVal) -> {
			if (newVal.intValue() == 42) {
				i1.set(17);
			}
		});
		i1.addListener(new Listener("Listener2: "));
		i1.addListener(new Listener("Listener3: "));
		i1.set(42);
	}

	private static class Listener implements ChangeListener<Number> {
		private final String label;
		private Listener(String label) { this.label = label; }

		@Override
		public void changed(ObservableValue<? extends Number> source, Number oldVal, Number newVal) {
			System.out.println(label + oldVal + " -> " + newVal);
		}
	}
}
