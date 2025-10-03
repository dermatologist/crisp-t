        # Visualization
        if visualize and corpus:
            click.echo("\n=== Generating Visualizations ===")
            _word_cloud = pathlib.Path(out) / "word_cloud.png" if out else None
            viz = QRVisualize(corpus=corpus, folder_path=str(_word_cloud))
            viz.plot_wordcloud()
            click.echo("✓ Word cloud generated")
            _word_freq = pathlib.Path(out) / "word_frequencies.png" if out else None
            viz.plot_frequency_distribution_of_words(
                folder_path=str(_word_freq), show=False
            )
            click.echo("✓ Word frequency distribution plot generated")