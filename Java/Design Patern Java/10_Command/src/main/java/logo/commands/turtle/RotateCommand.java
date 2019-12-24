package logo.commands.turtle;

import logo.Turtles;

public class RotateCommand implements TurtleCommand {
	private final Turtles turtles;
	private final int amount;

	public RotateCommand(Turtles turtles, int leftAmount) {
		this.turtles = turtles;
		this.amount = leftAmount;
	}

	@Override
	public void execute() {
		turtles.left(amount);
	}

	@Override
	public String toString() {
		return "Rotating " + amount + " degrees left.";
	}

}
