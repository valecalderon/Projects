namespace RentalQA.Core;

/// <summary>
/// Calculates rental pricing, including mileage overage and late fees.
/// Kept deliberately simple so the focus stays on TEST DESIGN, not business complexity.
/// </summary>
public class RentalPriceCalculator
{
    private const decimal MileageOverageFeePerMile = 0.79m;
    private const decimal LateFeePerDay = 25.00m;

    public decimal CalculateBasePrice(int days, decimal dailyRate)
    {
        if (days < 0)
            throw new ArgumentException("Rental days cannot be negative.", nameof(days));
        if (dailyRate < 0)
            throw new ArgumentException("Daily rate cannot be negative.", nameof(dailyRate));

        return days * dailyRate;
    }

    public decimal CalculateMileageOverageFee(int milesDriven, int milesIncluded)
    {
        if (milesDriven < 0 || milesIncluded < 0)
            throw new ArgumentException("Mileage values cannot be negative.");

        int overageMiles = milesDriven - milesIncluded;
        return overageMiles > 0 ? overageMiles * MileageOverageFeePerMile : 0m;
    }

    public decimal CalculateLateFee(DateTime dueDate, DateTime returnDate)
    {
        if (returnDate <= dueDate)
            return 0m;

        // Any partial day late counts as a full late day - a real business rule
        // that's easy to get wrong with naive date subtraction.
        int lateDays = (int)Math.Ceiling((returnDate - dueDate).TotalDays);
        return lateDays * LateFeePerDay;
    }

    public decimal CalculateTotalPrice(
        int days,
        decimal dailyRate,
        int milesDriven,
        int milesIncluded,
        DateTime dueDate,
        DateTime returnDate)
    {
        decimal total = CalculateBasePrice(days, dailyRate);
        total += CalculateMileageOverageFee(milesDriven, milesIncluded);
        total += CalculateLateFee(dueDate, returnDate);
        return total;
    }
}
