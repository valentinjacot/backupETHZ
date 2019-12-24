package Observer;

public class StockObserver implements Observer {
	private double ibmPrice;
	private double aaplPrice;
	private double googPrice;

	private static int observerIDTracker =0;
	private int ObserverId;
	
//	private Subject stockGraber; 
	
	public StockObserver(Subject stockGraber) {
//		this.stockGraber = stockGraber;
		this.ObserverId = ++observerIDTracker;
		System.out.println("New Observer " + this.ObserverId );
		
		stockGraber.register(this);
	}
	
	@Override
	public void update(double ibmPrice, double aaplPrice, double googPrice) {
		// TODO Auto-generated method stub
		this.ibmPrice =ibmPrice ;
		this.aaplPrice = aaplPrice;
		this.googPrice = googPrice;

		printThePrices();
	}
	public void printThePrices() {
		System.out.println("ObserverId: " +ObserverId + "\nIBM " + ibmPrice + "\nAapl " + aaplPrice+ "\ngoog" + googPrice);
	}

}
