using FluentAssertions;
using RentalQA.Core;
using Xunit;

namespace RentalQA.Tests;

public class RentalPriceCalculatorTests
{
    private readonly RentalPriceCalculator _calculator = new();

    // ---- Base price: standard cases + edge cases ----

    [Theory]
    [InlineData(3, 29.99, 89.97)]
    [InlineData(1, 50.00, 50.00)]
    public void CalculateBasePrice_StandardInputs_ReturnsCorrectTotal(int days, double rate, double expected)
    {
        var result = _calculator.CalculateBasePrice(days, (decimal)rate);
        result.Should().Be((decimal)expected);
    }

    [Fact]
    public void CalculateBasePrice_ZeroDays_ReturnsZero()
    {
        // Edge case: same-day return / cancelled rental should bill cleanly to zero, not error.
        _calculator.CalculateBasePrice(0, 29.99m).Should().Be(0m);
    }

    [Fact]
    public void CalculateBasePrice_NegativeDays_ThrowsArgumentException()
    {
        // Edge case: guards against an upstream date bug silently producing a refund.
        Action act = () => _calculator.CalculateBasePrice(-1, 29.99m);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void CalculateBasePrice_ZeroRate_ReturnsZero()
    {
        // Edge case: promotional/comped rental.
        _calculator.CalculateBasePrice(5, 0m).Should().Be(0m);
    }

    // ---- Mileage overage: this is the "hidden" business rule most candidates miss ----

    [Fact]
    public void CalculateMileageOverageFee_UnderIncludedMiles_ReturnsZero()
    {
        _calculator.CalculateMileageOverageFee(milesDriven: 80, milesIncluded: 100).Should().Be(0m);
    }

    [Fact]
    public void CalculateMileageOverageFee_ExactlyAtLimit_ReturnsZero()
    {
        // Boundary case: exactly at the included limit should not incur a fee.
        _calculator.CalculateMileageOverageFee(milesDriven: 100, milesIncluded: 100).Should().Be(0m);
    }

    [Fact]
    public void CalculateMileageOverageFee_OverLimit_ChargesPerMile()
    {
        // 120 driven, 100 included -> 20 miles over at $0.79/mile
        _calculator.CalculateMileageOverageFee(milesDriven: 120, milesIncluded: 100).Should().Be(15.80m);
    }

    // ---- Late fee: date math edge cases are where real bugs hide ----

    [Fact]
    public void CalculateLateFee_ReturnedOnTime_ReturnsZero()
    {
        var due = new DateTime(2026, 1, 10);
        var returned = new DateTime(2026, 1, 10);
        _calculator.CalculateLateFee(due, returned).Should().Be(0m);
    }

    [Fact]
    public void CalculateLateFee_ReturnedEarly_ReturnsZero()
    {
        var due = new DateTime(2026, 1, 10);
        var returned = new DateTime(2026, 1, 8);
        _calculator.CalculateLateFee(due, returned).Should().Be(0m);
    }

    [Fact]
    public void CalculateLateFee_PartialDayLate_ChargesFullDay()
    {
        // Edge case: returned just 3 hours late - business rule says this still
        // counts as one full late day. Naive (returned - due).Days would return 0 and miss this.
        var due = new DateTime(2026, 1, 10, 10, 0, 0);
        var returned = new DateTime(2026, 1, 10, 13, 0, 0);
        _calculator.CalculateLateFee(due, returned).Should().Be(25.00m);
    }

    [Fact]
    public void CalculateLateFee_TwoFullDaysLate_ChargesTwoDayFees()
    {
        var due = new DateTime(2026, 1, 10);
        var returned = new DateTime(2026, 1, 12);
        _calculator.CalculateLateFee(due, returned).Should().Be(50.00m);
    }
}
