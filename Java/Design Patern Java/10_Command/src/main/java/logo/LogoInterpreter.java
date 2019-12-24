package logo;

import java.awt.Color;
import java.util.Scanner;

import logo.commands.Command;
import logo.commands.control.BeginMacroCommand;
import logo.commands.control.EndMacroCommand;
import logo.commands.control.ErrorCommand;
import logo.commands.control.ExitCommand;
import logo.commands.control.NotFoundCommand;
import logo.commands.control.RedoCommand;
import logo.commands.control.UndoCommand;
import logo.commands.turtle.ClearScreenCommand;
import logo.commands.turtle.MoveCommand;
import logo.commands.turtle.PenDownCommand;
import logo.commands.turtle.PenUpCommand;
import logo.commands.turtle.RepeatCommand;
import logo.commands.turtle.RotateCommand;
import logo.commands.turtle.TurtleCommand;
import logo.parser.CommandRegistry;
import logo.parser.Parser;

public class LogoInterpreter {
	private final Turtles turtles;

	private final HistoryManager historyManager;

	private final MacroManager macroManager;

	public HistoryManager getHistoryManager() {
		return historyManager;
	}

	private Parser parser;

	public LogoInterpreter() {
		turtles = new Turtles();
		historyManager = new HistoryManagerImpl(this);
		macroManager = new MacroManagerImpl(this);
		System.out.println("Starting interpreter...");
	}

	public static void main(final String[] args) {
		new LogoInterpreter().run();
	}

	private boolean running;

	public void stop() {
		running = false;
	}

	public void repaint() {
		resetTurtle();
		turtles.setColor(Color.BLACK);
		for(Command c : historyManager.getCommands()) {
			c.execute();
		}
	}
	
	public void setColor(Color c) {
		turtles.setColor(c);
	}
	
	public void run() {
		turtles.show();

		initializeParser();
		resetTurtle();
		Scanner scanner = new Scanner(System.in);
		running = true;
		while (running) {
			Command command = parser.parse(scanner);
			if (macroManager.isRecordingMacro()) {
				if(command instanceof TurtleCommand) {
					macroManager.addCommand((TurtleCommand)command);
				} else if(command instanceof EndMacroCommand) {
					command.execute();
				} else {
					System.out.println("This command cannot be used in a macro!");
				}
			} else {
				if (command instanceof TurtleCommand) {
					historyManager.addCommand((TurtleCommand)command);
				}
				System.out.println("" + command);
				command.execute();
			}
		}
		scanner.close();
		turtles.quit();
	}

	private void initializeParser() {
		CommandRegistry commandRegistry = new CommandRegistry();
		commandRegistry.registerCommand("backward", scanner -> {
			int backward = scanner.nextInt();
			return new MoveCommand(turtles, -backward);
		});
		commandRegistry.registerCommand("forward", scanner -> {
			int forward = scanner.nextInt();
			return new MoveCommand(turtles, forward);
		});
		commandRegistry.registerCommand("move", scanner -> {
			int forward = scanner.nextInt();
			return new MoveCommand(turtles, forward);
		});
		commandRegistry.registerCommand("left", scanner -> {
			int left = scanner.nextInt();
			return new RotateCommand(turtles, left);
		});
		commandRegistry.registerCommand("right", scanner -> {
			int right = scanner.nextInt();
			return new RotateCommand(turtles, -right);
		});
		commandRegistry.registerCommand("exit", scanner -> {
			return new ExitCommand(this);
		});
		commandRegistry.registerCommand("clearscreen", scanner -> {
			return new ClearScreenCommand(this);
		});
		commandRegistry.registerCommand("penup", scannner -> {
			return new PenUpCommand(turtles);
		});
		commandRegistry.registerCommand("pendown", scannner -> {
			return new PenDownCommand(turtles);
		});
		commandRegistry.registerCommand("repeat", scanner -> {
			int amount = scanner.nextInt();
			Command cmd = parser.parse(scanner);
			if(cmd instanceof TurtleCommand) {
				return new RepeatCommand(amount, (TurtleCommand)cmd);
			} else {
				return new ErrorCommand("Turtle Command expected.");
			}
		});
		commandRegistry.registerCommand("undo", scanner -> {
			return new UndoCommand(historyManager);
		});
		commandRegistry.registerCommand("redo", scanner -> {
			return new RedoCommand(historyManager);
		});
		commandRegistry.registerCommand("macrorecord", scanner -> {
			String name = scanner.next();
			return new BeginMacroCommand(name, macroManager);
		});
		commandRegistry.registerCommand("macrosave", scanner -> {
			return new EndMacroCommand(macroManager);
		});
		commandRegistry.registerCommand("macrorun", scanner -> {
			String name = scanner.next();
			try {
				return macroManager.getCommand(name);
			} catch (Exception e) {
				e.printStackTrace();
				// return relatively gracefully
				return new NotFoundCommand("");
			}
		});

		parser = new Parser(commandRegistry);
	}

	public void resetTurtle() {
		turtles.moveTo(200, 200);
		turtles.clear();
		turtles.setDirection(90);
		turtles.down();
	}
}
