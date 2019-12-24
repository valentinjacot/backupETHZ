package logo.commands;

import java.util.Scanner;

public interface CommandFactory {
	Command createCommand(Scanner scanner);
}