package patterns.builder;

public class Pizza {

	private final int size;
	private final boolean cheese, pepperoni, bacon;

	public static Builder builder(int size) {
		return new Builder(size);
	}

	public static class Builder {
		private final int size; // required
		private boolean cheese = false; // optional
		private boolean pepperoni = false; // optional
		private boolean bacon = false; // optional

		private Builder(int size) {
			this.size = size;
		}

		public Builder cheese(boolean c) {
			cheese = c;
			return this;
		}

		public Builder pepperoni(boolean p) {
			pepperoni = p;
			return this;
		}

		public Builder bacon(boolean b) {
			bacon = b;
			return this;
		}

		public Pizza build() {
			return new Pizza(this);
		}
	}

	private Pizza(Builder builder) {
		size = builder.size;
		cheese = builder.cheese;
		pepperoni = builder.pepperoni;
		bacon = builder.bacon;
	}
	
	public int getSize() {
		return size;
	}

	public boolean isCheese() {
		return cheese;
	}

	public boolean isPepperoni() {
		return pepperoni;
	}

	public boolean isBacon() {
		return bacon;
	}

}
