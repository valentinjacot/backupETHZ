package Observer;

import java.util.ArrayList;

public class StockGraber implements Subject {
	private ArrayList<Observer> observers;
	private double ibmPrice;
	private double aaplPrice;
	private double googPrice;
	
	public StockGraber() {
		observers = new ArrayList<Observer>();
	}
	
	@Override
	public void register(Observer o) {
		observers.add(o);

	}

	@Override
	public void unregister(Observer o) {
		int observerIndex = observers.indexOf(o);
		System.out.println("Observer " + (observerIndex + 1) + " deleted");
		observers.remove(observerIndex);
	}

	@Override
	public void notifyListeners() {
		for (Observer o:observers) {
			o.update(ibmPrice,aaplPrice, googPrice);
		}
	}

	public void setIbmPrice(double ibmPrice) {
		this.ibmPrice = ibmPrice;
		notifyListeners();
	}

	public void setAaplPrice(double aaplPrice) {
		this.aaplPrice = aaplPrice;
		notifyListeners();
	}

	public void setGoogPrice(double googPrice) {
		this.googPrice = googPrice;
		notifyListeners();
	}
	
	

}
