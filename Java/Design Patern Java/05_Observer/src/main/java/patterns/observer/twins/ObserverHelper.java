package patterns.observer.twins;

import java.util.Observable;

@SuppressWarnings("deprecation")
public class ObserverHelper<T> extends Observable {
	private final T observableModel;

	public ObserverHelper(T observableModel) {
		this.observableModel = observableModel;
	}

	public T getObservableModel() {
		return observableModel;
	}

	// lift visibility into this package
	protected void setChanged() {
		super.setChanged();
	}

	protected void clearChanged() {
		super.clearChanged();
	}
}
