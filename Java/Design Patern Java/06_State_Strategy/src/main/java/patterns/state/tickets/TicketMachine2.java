package patterns.state.tickets;

public class TicketMachine2 implements TicketMachine {
	private int destination;
	private boolean firstClass, retour, halfPrice;
	private double price;
	private double enteredMoney;

	private interface State {
		void setDestination(int destination);
		void setFirstClass(boolean firstClass);
		void setReturnTicket(boolean retour) ;
		void setHalfPrice(boolean halfPrice);
		void enterMoney(double amount);
		void cancel();
	}

	private final State INIT = new StateInit();
	private final State DEST_SELECTED = new StateDestSelected();
	private final State MONEY_ENTERED = new StateMoneyEntered();
	
	private State state = INIT;
	
	private abstract class AbstractState implements State {
		@Override public void setDestination(int destination) { throw new IllegalStateException(); }
		@Override public void setFirstClass(boolean firstClass) { throw new IllegalStateException(); }
		@Override public void setReturnTicket(boolean retour) { throw new IllegalStateException(); }
		@Override public void setHalfPrice(boolean halfPrice) { throw new IllegalStateException(); }
		@Override public void enterMoney(double amount) { throw new IllegalStateException(); }
		@Override public void cancel() { state = INIT; }
	}

	private class StateInit extends AbstractState {
		@Override 
		public void setDestination(int dest) {
			destination = dest;
			firstClass = false;
			halfPrice = false;
			firstClass = false;
			enteredMoney = 0;
			price = calculatePrice(destination, firstClass, retour, halfPrice);
			state = DEST_SELECTED;
		}
	}
	
	private class StateDestSelected extends AbstractState {
		@Override
		public void setFirstClass(boolean fc) {
			firstClass = fc;
			price = calculatePrice(destination, firstClass, retour, halfPrice);
		}

		@Override
		public void setReturnTicket(boolean rt) {
			retour = rt;
			price = calculatePrice(destination, firstClass, retour, halfPrice);
		}

		@Override
		public void setHalfPrice(boolean hp) {
			halfPrice = hp;
			price = calculatePrice(destination, firstClass, retour, halfPrice);
		}

		@Override
		public void enterMoney(double amount) {
			state = MONEY_ENTERED;
			state.enterMoney(amount);
		}
	}

	private class StateMoneyEntered extends AbstractState {
		@Override
		public void enterMoney(double amount) {
			enteredMoney += amount;
			if (enteredMoney >= price) {
				printTicket(destination, price, firstClass, retour, halfPrice);
				System.out.printf("Change: %4.2f\n", enteredMoney-price);
				state = INIT;
			}
		}
		
		@Override
		public void cancel() {
			System.out.printf("Change: %4.2f\n", enteredMoney);
			super.cancel();
		}
	}
	
	@Override public void setDestination(int destination) { state.setDestination(destination); }
	@Override public void setFirstClass(boolean firstClass) { state.setFirstClass(firstClass); }
	@Override public void setReturnTicket(boolean retour) { state.setReturnTicket(retour); }
	@Override public void setHalfPrice(boolean halfPrice) { state.setHalfPrice(halfPrice); }
	@Override public void enterMoney(double amount) { state.enterMoney(amount); }
	@Override public void cancel() { state.cancel(); }

	private void printTicket(int destination, double price, boolean firstClass,
			boolean retour, boolean halfPrice) {
		System.out.printf("Ticket to %s %s%s%s for CHF %4.2f\n", destination, 
				firstClass ? "1Kl" :  "2Kl",
				retour ? " retour" : "",
				halfPrice ? " [ermaessigt]": "", price);
	}

	private double calculatePrice(int destination, boolean firstClass,
			boolean retour, boolean halfPrice) {
		double price = Math.floor(destination / 100.0 * 20) / 20;
		if(firstClass) price = 2 * price;
		if(retour) price = 2 * price;
		if(halfPrice) price = price / 2;
		return price;
	}

	@Override
	public double getPrice() {
		return price;
	}
	
	@Override
	public double getEnteredMoney() {
		return enteredMoney;
	}
	
	@Override
	public boolean isFirstClass() {
		return firstClass;
	}

	@Override
	public boolean isRetour() {
		return retour;
	}

	@Override
	public boolean isHalfPrice() {
		return halfPrice;
	}

	@Override
	public boolean isInStateInit() {
		return state == INIT;
	}

	@Override
	public boolean isInStateDestSelected() {
		return state == DEST_SELECTED;
	}

	@Override
	public boolean isInStateMoneyEntered() {
		return state == MONEY_ENTERED;
	}

}
