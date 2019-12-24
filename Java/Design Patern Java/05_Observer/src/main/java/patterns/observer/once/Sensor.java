package patterns.observer.once;


public abstract class Sensor implements Observable {
	private double value;

	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
		notifyObservers();
	}

}
