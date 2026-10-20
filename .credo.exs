%{
  configs: [
    %{
      name: "default",
      files: %{
        included: ["lib/"]
      },
      strict: true,
      checks: [
        # Credo deprecated this check; the project has no rule on `Enum.into/2` against `Map.new/2`.
        {Credo.Check.Refactor.MapInto, false},
        # `Logger` macros already skip argument work for levels that are off.
        {Credo.Check.Warning.LazyLogging, false},
        {Credo.Check.Readability.LargeNumbers, only_greater_than: 86400},
        {Credo.Check.Readability.ParenthesesOnZeroArityDefs, parens: true},
        {Credo.Check.Readability.Specs, tags: []},
        {Credo.Check.Readability.StrictModuleLayout, tags: []}
      ]
    }
  ]
}
