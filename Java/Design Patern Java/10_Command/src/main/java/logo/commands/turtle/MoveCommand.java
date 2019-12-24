package logo.commands.turtle;

import logo.Turtles;

public class MoveCommand implements TurtleCommand {
	private final Turtles turtles;
	private final int amount;

	public MoveCommand(Turtles turtles, int amount) {
		this.turtles = turtles;
		this.amount = amount;
	}

	@Override
	public void execute() {
		turtles.move(amount);
	}

	@Override
	public String toString() {
		return "Moving " + amount + " steps.";
	}

}
