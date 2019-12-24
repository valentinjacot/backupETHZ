package patterns.clone.company.clone;

public class Employee implements Cloneable {
	private String name;
	private int yearOfBirth;

	public Employee(String name, int yearOfBirth) {
		this.name = name;
		this.yearOfBirth = yearOfBirth;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getYearOfBirth() {
		return yearOfBirth;
	}

	public void setYearOfBirth(int yearOfBirth) {
		this.yearOfBirth = yearOfBirth;
	}

	@Override
	public boolean equals(Object o) {
		if (o instanceof Employee) {
			Employee p = (Employee) o;
			return (p.yearOfBirth == yearOfBirth) && (p.name.equals(name));
		}
		return false;
	}

	@Override
	public Employee clone() {
		try {
			return (Employee) super.clone();
		} catch (CloneNotSupportedException e) {
			throw new InternalError();
		}
	}
}
