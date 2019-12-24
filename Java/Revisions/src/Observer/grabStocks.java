package Observer;

public class grabStocks {

	public static void main(String[] args) {
		
		StockGraber stockGraber = new StockGraber();
		
		StockObserver obs1 = new StockObserver(stockGraber);
//		
//		stockGraber.setAaplPrice(123.0);
//		stockGraber.setGoogPrice(145.8);
//		stockGraber.setIbmPrice(189.6);
		StockObserver obs2 = new StockObserver(stockGraber);
		
		stockGraber.setAaplPrice(123.0);
		stockGraber.setGoogPrice(145.8);
		stockGraber.setIbmPrice(189.6);
		
		
	}

}
