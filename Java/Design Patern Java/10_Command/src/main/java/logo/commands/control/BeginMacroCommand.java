package logo.commands.control;

import logo.MacroManager;

public class BeginMacroCommand extends AbstractMacroCommand {
	private final String macroName;

	public BeginMacroCommand(String macroName, MacroManager macroManager) {
		super(macroManager);
		this.macroName = macroName;
	}

	@Override
	public void execute() {
		getMacroManager().startMacro(macroName);
	}

	@Override
	public String toString() {
		return "Begin macro " + macroName + ".";
	}

}
