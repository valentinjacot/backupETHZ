package patterns.state.tickets.fx;

// Interface used by the TicketMachine GUI
public interface TicketMachine {
	
	// user events (performed at the ticket machine)
	void setDestination(int destination);

	void setFirstClass(boolean firstClass);
	void setDayTicket(boolean retour);
	void setHalfPrice(boolean halfPrice);

	void enterMoney(double amount);
	void cancel();

	// query methods (display on the ticket machine)
	double getPrice();
	double getEnteredMoney();

	boolean isFirstClass();
	boolean isDayTicket();
	boolean isHalfPrice();
	
	boolean isInStateInit();
	boolean isInStateDestSelected();
	boolean isInStateMoneyEntered();
}